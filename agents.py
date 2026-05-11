"""LLM-driven Pokémon battle agents.

Adapted from the Hugging Face Agents Course (Bonus Unit 3) reference at
https://huggingface.co/spaces/Jofthomas/twitch_streaming/blob/main/agents.py
"""
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

# --- Google Gemini ---
from google import genai
from google.genai import types
from google.genai.errors import ClientError

# --- OpenAI-compatible (LM Studio) ---
from openai import AsyncOpenAI

# --- Poke-Env ---
from poke_env.data import GenData
from poke_env.player import Player

# Type chart used for damage-multiplier calculations injected into the prompt.
# Gen 9 covers the most common Showdown ladder formats; falls back gracefully
# when a battle uses a different generation.
_TYPE_CHART = GenData.from_gen(9).type_chart


def normalize_name(name: str) -> str:
    """Lowercase and remove non-alphanumeric characters."""
    return "".join(filter(str.isalnum, name)).lower()


_ANALYSIS_PROP = {
    "analysis": {
        "type": "string",
        "description": (
            "Step-by-step analysis (3-5 sentences) BEFORE choosing. Cover: "
            "(1) type matchup of your active vs opponent, "
            "(2) speed comparison and what the opponent is likely to do, "
            "(3) HP / status situation and threats, "
            "(4) which of your options (move OR switch) maximizes expected value and why."
        ),
    }
}

_REASONING_PROP = {
    "reasoning": {
        "type": "string",
        "description": "One short sentence (under 250 chars) summarizing the choice. Will be sent as a chat message in the battle room.",
    }
}

STANDARD_TOOL_SCHEMA = {
    "choose_move": {
        "name": "choose_move",
        "description": "Selects and executes an available attacking or status move. The analysis field is required and must be filled FIRST.",
        "parameters": {
            "type": "object",
            # Order matters: analysis listed first encourages the model to write
            # its reasoning before committing to an action.
            "properties": {
                **_ANALYSIS_PROP,
                "move_name": {
                    "type": "string",
                    "description": "The exact name or ID (e.g., 'thunderbolt', 'swordsdance') of the move to use. Must be one of the available moves.",
                },
                **_REASONING_PROP,
            },
            "required": ["analysis", "move_name"],
        },
    },
    "choose_switch": {
        "name": "choose_switch",
        "description": "Selects an available Pokémon from the bench to switch into. The analysis field is required and must be filled FIRST.",
        "parameters": {
            "type": "object",
            "properties": {
                **_ANALYSIS_PROP,
                "pokemon_name": {
                    "type": "string",
                    "description": "The exact name of the Pokémon species to switch to (e.g., 'Pikachu', 'Charizard'). Must be one of the available switches.",
                },
                **_REASONING_PROP,
            },
            "required": ["analysis", "pokemon_name"],
        },
    },
}

CHAT_MAX_LEN = 280


class LLMAgentBase(Player):
    # Showdown random-battle timer model (approximate):
    #   * Starting bank: TIMER_BANK_INITIAL seconds
    #   * Per-turn bonus: TIMER_PER_TURN added to bank each turn (capped at
    #     TIMER_BANK_CAP)
    #   * If your bank hits 0 you lose the battle on time.
    # We track the bank locally and shrink decision_timeout as it drains. With
    # this safety net we should never run out of time even with a slow local LLM.
    TIMER_BANK_INITIAL: float = 150.0
    TIMER_PER_TURN: float = 10.0
    TIMER_BANK_CAP: float = 200.0
    TIMER_SAFETY_MARGIN: float = 12.0  # never let the bank get this close to 0
    decision_timeout: float = 60.0  # baseline; shrunk dynamically based on bank

    # File where every turn's full analysis + reasoning is appended. Lets you
    # `tail -f battle_log.txt` (or open in VS Code with auto-reload) to watch the
    # agent's thinking, since Showdown's --no-security mode silently drops chat
    # from guest accounts in battle rooms.
    battle_log_path: Path = Path("battle_log.txt")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.standard_tools = STANDARD_TOOL_SCHEMA
        # battle_tag -> list of turn records (each: dict with turn, action, summary)
        self.battle_history: Dict[str, list] = {}
        # battle_tag -> remaining seconds in the Showdown timer bank (estimate)
        self._timer_bank: Dict[str, float] = {}

    def _append_battle_log(
        self,
        battle,
        action_text: str,
        analysis: Optional[str],
        reasoning: Optional[str],
        is_first_turn: bool,
    ) -> None:
        # Truncate the log at the start of every new match so it only ever
        # holds the current battle.
        mode = "w" if is_first_turn else "a"
        try:
            with self.battle_log_path.open(mode, encoding="utf-8") as f:
                if is_first_turn:
                    f.write(
                        f"{'=' * 70}\n"
                        f"NEW BATTLE: {battle.battle_tag}\n"
                        f"{'=' * 70}\n"
                    )
                f.write(f"\n[T{battle.turn}] {action_text}\n")
                if analysis:
                    f.write(f"  Analysis: {analysis.strip()}\n")
                if reasoning:
                    f.write(f"  Chat:     {reasoning.strip()}\n")
                f.flush()
        except Exception as e:
            print(f"Failed to write battle_log.txt: {e}")

    def _record_turn(
        self,
        battle,
        action_text: str,
        analysis: Optional[str],
        reasoning: Optional[str] = None,
    ) -> None:
        log = self.battle_history.setdefault(battle.battle_tag, [])
        is_first_turn = len(log) == 0
        # Keep only the first sentence of the analysis to avoid blowing up the prompt.
        summary = ""
        if analysis:
            summary = re.split(r"(?<=[.!?])\s+", analysis.strip(), maxsplit=1)[0][:240]
        log.append(
            {
                "turn": battle.turn,
                "action": action_text,
                "summary": summary,
            }
        )
        self._append_battle_log(battle, action_text, analysis, reasoning, is_first_turn)

    def _format_history(self, battle) -> str:
        log = self.battle_history.get(battle.battle_tag, [])
        if not log:
            return ""
        lines = [f"- T{rec['turn']}: {rec['action']} — {rec['summary']}" for rec in log]
        return "Battle history (your past actions this match):\n" + "\n".join(lines)

    @staticmethod
    def _describe_move(move) -> str:
        """Build a short effect description from poke-env's structured Move data
        so the LLM doesn't have to recall what each move does from memory."""
        parts = []
        if move.priority > 0:
            parts.append(f"+{move.priority} priority (moves first)")
        elif move.priority < 0:
            parts.append(f"{move.priority} priority (moves last)")

        if move.heal and move.heal > 0:
            parts.append(f"heals user {int(move.heal * 100)}%HP")
        elif "heal" in (move.flags or set()) and move.base_power == 0:
            parts.append("heals user (amount varies by weather)")

        if getattr(move, "drain", 0) and move.drain > 0:
            parts.append(f"drains {int(move.drain * 100)}% of damage dealt as HP")

        if getattr(move, "recoil", 0) and move.recoil > 0:
            parts.append(f"recoil: user takes {int(move.recoil * 100)}% of damage dealt")

        if move.boosts:
            target_attr = getattr(move, "target", None)
            target_name = getattr(target_attr, "name", "") if target_attr else ""
            target_label = "user" if target_name == "SELF" else "target"
            for stat, val in move.boosts.items():
                sign = "+" if val > 0 else ""
                parts.append(f"{sign}{val} {stat} on {target_label}")

        if move.self_boost:
            for stat, val in move.self_boost.items():
                sign = "+" if val > 0 else ""
                parts.append(f"self {sign}{val} {stat}")

        if move.status:
            parts.append(f"inflicts {move.status.name}")

        for sec in (move.secondary or []):
            chance = sec.get("chance", "?")
            if "status" in sec:
                parts.append(f"{chance}% chance: inflict {sec['status']}")
            elif "volatileStatus" in sec:
                parts.append(f"{chance}% chance: {sec['volatileStatus']}")
            elif "boosts" in sec:
                for s, v in sec["boosts"].items():
                    sign = "+" if v > 0 else ""
                    parts.append(f"{chance}% chance: {sign}{v} {s} on target")
            elif "self" in sec and isinstance(sec["self"], dict) and "boosts" in sec["self"]:
                for s, v in sec["self"]["boosts"].items():
                    sign = "+" if v > 0 else ""
                    parts.append(f"{chance}% chance: self {sign}{v} {s}")

        return "; ".join(parts)

    @staticmethod
    def _move_effectiveness(move, target) -> Optional[float]:
        """Type multiplier of `move` against `target` Pokemon, or None if not applicable."""
        if target is None or move.base_power <= 0:
            return None
        try:
            t1, t2 = (target.types + [None, None])[:2]
            return move.type.damage_multiplier(t1, t2, type_chart=_TYPE_CHART)
        except Exception:
            return None

    @staticmethod
    def _estimate_damage_pct(move, attacker, target) -> Optional[float]:
        """Rough % of target's max HP this move would deal. Random-battle assumptions
        (level 100, 84 EVs, 31 IVs, neutral nature). Ignores items, abilities, crit,
        weather, screens — gives the model a ballpark, not a calc-mon prediction."""
        if target is None or move.base_power <= 0:
            return None
        try:
            level = 100
            cat = move.category.name
            atk_key, def_key = ("atk", "def") if cat == "PHYSICAL" else ("spa", "spd")
            atk_base = attacker.base_stats.get(atk_key)
            def_base = target.base_stats.get(def_key)
            hp_base = target.base_stats.get("hp")
            if not (atk_base and def_base and hp_base):
                return None

            def stat(base):
                return int(((2 * base + 31 + 21) * level) / 100 + 5)

            def hp_stat(base):
                return int(((2 * base + 31 + 21) * level) / 100 + level + 10)

            atk = stat(atk_base)
            defense = stat(def_base)
            target_hp = hp_stat(hp_base)

            base_dmg = ((((2 * level) / 5 + 2) * move.base_power * atk / defense) / 50 + 2)
            t1, t2 = (target.types + [None, None])[:2]
            eff = move.type.damage_multiplier(t1, t2, type_chart=_TYPE_CHART)
            stab = 1.5 if move.type in attacker.types else 1.0
            damage = base_dmg * eff * stab * 0.925  # avg of 0.85-1.0 random factor
            return (damage / target_hp) * 100
        except Exception:
            return None

    @staticmethod
    def _defensive_profile(pkmn, attacker_types) -> str:
        """How much each of the attacker's STAB types would hit `pkmn` for."""
        parts = []
        for atype in attacker_types:
            if atype is None:
                continue
            try:
                t1, t2 = (pkmn.types + [None, None])[:2]
                mult = atype.damage_multiplier(t1, t2, type_chart=_TYPE_CHART)
                parts.append(f"vs {str(atype).lower()}: {mult:g}x")
            except Exception:
                continue
        return ", ".join(parts) if parts else "n/a"

    @staticmethod
    def _modifier_notes(pkmn) -> str:
        """Describes visible speed modifiers (boosts, paralysis, known item)."""
        notes = []
        boost = (pkmn.boosts or {}).get("spe", 0)
        if boost:
            mult = (2 + boost) / 2 if boost >= 0 else 2 / (2 - boost)
            notes.append(f"{boost:+d} stage x{mult:g}")
        if pkmn.status and getattr(pkmn.status, "name", "") == "PAR":
            notes.append("paralyzed x0.5")
        if str(pkmn.item or "").lower() == "choicescarf":
            notes.append("Choice Scarf x1.5")
        return ", ".join(notes)

    @staticmethod
    def _own_effective_speed(active) -> int:
        """Exact effective speed for our own active Pokemon."""
        base = (active.stats or {}).get("spe") or active.base_stats.get("spe", 0) * 2
        boost = (active.boosts or {}).get("spe", 0)
        mult = (2 + boost) / 2 if boost >= 0 else 2 / (2 - boost)
        if active.status and getattr(active.status, "name", "") == "PAR":
            mult *= 0.5
        if str(active.item or "").lower() == "choicescarf":
            mult *= 1.5
        return int(base * mult)

    @staticmethod
    def _opponent_speed_range(opponent, level: int = 84):
        """Min / max plausible randbat speed for the opponent.
        Min: 85 EVs, neutral nature. Max: 252 EVs, beneficial nature.
        Both apply known visible modifiers (boosts, paralysis, Choice Scarf if seen).
        """
        b = opponent.base_stats.get("spe", 0)
        # randbat: 31 IVs, level 84 default
        min_stat = int(((2 * b + 31 + 85 // 4) * level) / 100 + 5)
        max_stat = int((((2 * b + 31 + 252 // 4) * level) / 100 + 5) * 1.1)

        boost = (opponent.boosts or {}).get("spe", 0)
        mult = (2 + boost) / 2 if boost >= 0 else 2 / (2 - boost)
        if opponent.status and getattr(opponent.status, "name", "") == "PAR":
            mult *= 0.5
        if str(opponent.item or "").lower() == "choicescarf":
            mult *= 1.5
        return int(min_stat * mult), int(max_stat * mult)

    @staticmethod
    def _speed_comparison(active, opponent) -> str:
        if opponent is None:
            return "unknown"
        try:
            mine = LLMAgentBase._own_effective_speed(active)
            opp_min, opp_max = LLMAgentBase._opponent_speed_range(opponent)
            my_notes = LLMAgentBase._modifier_notes(active)
            opp_notes = LLMAgentBase._modifier_notes(opponent)
            mine_str = f"{mine}" + (f" ({my_notes})" if my_notes else "")
            opp_str = f"~{opp_min}-{opp_max} range" + (f" ({opp_notes})" if opp_notes else "")
            if mine > opp_max:
                verdict = "yes (you outspeed even the fastest realistic spread)"
            elif mine < opp_min:
                verdict = "no (opponent outspeeds even at slowest spread)"
            else:
                verdict = "UNCERTAIN — depends on opp's hidden EV/nature spread"
            warn = " — opp's item/ability may also flip this (Scarf, Sand Rush, Tailwind, etc.)"
            return f"{verdict}. Yours: {mine_str}, Opp: {opp_str}.{warn}"
        except Exception:
            return "unknown"

    def _format_battle_state(self, battle) -> str:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        active_info = (
            f"Your active Pokemon: {active.species} "
            f"(Type: {'/'.join(map(str, active.types))}) "
            f"HP: {active.current_hp_fraction * 100:.1f}% "
            f"Status: {active.status.name if active.status else 'None'} "
            f"Boosts: {active.boosts}"
        )

        if opponent:
            opp_info_str = (
                f"{opponent.species} "
                f"(Type: {'/'.join(map(str, opponent.types))}) "
                f"HP: {opponent.current_hp_fraction * 100:.1f}% "
                f"Status: {opponent.status.name if opponent.status else 'None'} "
                f"Boosts: {opponent.boosts}"
            )
            speed_note = self._speed_comparison(active, opponent)
            opponent_info = (
                f"Opponent's active Pokemon: {opp_info_str}\n"
                f"You outspeed opponent: {speed_note}"
            )
        else:
            opponent_info = "Opponent's active Pokemon: Unknown"

        moves_info = "Available moves:\n"
        if battle.available_moves:
            lines = []
            for move in battle.available_moves:
                cat = move.category.name
                desc = self._describe_move(move)
                desc_note = f" | Effect: {desc}" if desc else ""
                if move.base_power <= 0:
                    # Status moves: make it loud that they don't deal damage so the
                    # model doesn't reason about type effectiveness on them.
                    lines.append(
                        f"- {move.id} | Type: {move.type} | STATUS (no damage, side effect only) "
                        f"| Acc: {move.accuracy} | PP: {move.current_pp}/{move.max_pp}{desc_note}"
                    )
                else:
                    eff = self._move_effectiveness(move, opponent)
                    dmg = self._estimate_damage_pct(move, active, opponent)
                    eff_note = f" | Eff: {eff:g}x" if eff is not None else ""
                    dmg_note = f" | Est dmg: ~{dmg:.0f}% of opp HP" if dmg is not None else ""
                    lines.append(
                        f"- {move.id} | Type: {move.type} | {cat} | BP: {move.base_power} "
                        f"| Acc: {move.accuracy} | PP: {move.current_pp}/{move.max_pp}"
                        f"{eff_note}{dmg_note}{desc_note}"
                    )
            moves_info += "\n".join(lines)
        else:
            moves_info += "- None (Must switch or Struggle)"

        switches_info = "Available switches:\n"
        if battle.available_switches:
            opp_types = opponent.types if opponent else []
            lines = []
            for pkmn in battle.available_switches:
                profile = (
                    self._defensive_profile(pkmn, opp_types) if opp_types else "n/a"
                )
                lines.append(
                    f"- {pkmn.species} (HP: {pkmn.current_hp_fraction * 100:.1f}%, "
                    f"Status: {pkmn.status.name if pkmn.status else 'None'}, "
                    f"Type: {'/'.join(map(str, pkmn.types))}, "
                    f"Defenses {profile})"
                )
            switches_info += "\n".join(lines)
        else:
            switches_info += "- None"

        history_block = self._format_history(battle)
        state_str = (
            (history_block + "\n\n" if history_block else "")
            + f"{active_info}\n"
            f"{opponent_info}\n\n"
            f"{moves_info}\n\n"
            f"{switches_info}\n\n"
            f"Weather: {battle.weather}\n"
            f"Terrains: {battle.fields}\n"
            f"Your Side Conditions: {battle.side_conditions}\n"
            f"Opponent Side Conditions: {battle.opponent_side_conditions}"
        )
        return state_str.strip()

    def _find_move_by_name(self, battle, move_name: str):
        normalized_name = normalize_name(move_name)
        for move in battle.available_moves:
            if move.id == normalized_name:
                return move
        # Fallback: substring match against IDs (handles "earth power" -> "earthpower"
        # variants that normalize_name should catch but also LLM verbosity like
        # "use Flamethrower"). poke-env Move has no .name attribute, only .id.
        for move in battle.available_moves:
            if normalized_name in move.id or move.id in normalized_name:
                print(
                    f"Warning: Loose-matched move id '{move.id}' from input '{move_name}'."
                )
                return move
        return None

    def _find_pokemon_by_name(self, battle, pokemon_name: str):
        normalized_name = normalize_name(pokemon_name)
        for pkmn in battle.available_switches:
            if normalize_name(pkmn.species) == normalized_name:
                return pkmn
        return None

    async def _send_reasoning(self, battle, reasoning: Optional[str]) -> None:
        if not reasoning:
            print("DEBUG send_reasoning: no reasoning provided")
            return
        text = reasoning.strip().replace("\n", " ")[:CHAT_MAX_LEN]
        print(f"DEBUG send_reasoning: room={battle.battle_tag!r} text={text!r}")
        try:
            await self.ps_client.send_message(text, room=battle.battle_tag)
            print("DEBUG send_reasoning: send_message returned without error")
        except Exception as e:
            print(f"Failed to send reasoning to chat: {e}")

    def _timer_budget_for(self, battle) -> float:
        """How many seconds we can afford to spend on the LLM this turn."""
        bank = self._timer_bank.setdefault(battle.battle_tag, self.TIMER_BANK_INITIAL)
        # Subtract safety margin so we always finish with cushion to spare.
        usable = bank - self.TIMER_SAFETY_MARGIN
        # Never spend more than baseline decision_timeout in a single turn.
        return max(0.0, min(self.decision_timeout, usable))

    def _consume_timer(self, battle, elapsed: float) -> None:
        """Update the bank after a turn (subtract elapsed, add per-turn bonus)."""
        tag = battle.battle_tag
        new_bank = self._timer_bank.get(tag, self.TIMER_BANK_INITIAL) - elapsed + self.TIMER_PER_TURN
        self._timer_bank[tag] = max(0.0, min(self.TIMER_BANK_CAP, new_bank))

    def _emergency_pick(self, battle):
        """Deterministic fallback when the timer bank is critically low. Picks
        the move with highest estimated damage; switches if no moves available."""
        opponent = battle.opponent_active_pokemon
        if battle.available_moves:
            scored = []
            for m in battle.available_moves:
                dmg = self._estimate_damage_pct(m, battle.active_pokemon, opponent) or 0
                scored.append((dmg, m))
            scored.sort(reverse=True, key=lambda x: x[0])
            return self.create_order(scored[0][1])
        if battle.available_switches:
            return self.choose_random_move(battle)
        return self.choose_default_move(battle)

    async def choose_move(self, battle) -> str:
        battle_state_str = self._format_battle_state(battle)
        budget = self._timer_budget_for(battle)
        bank = self._timer_bank[battle.battle_tag]
        print(f"Timer: bank≈{bank:.0f}s, budget for LLM this turn={budget:.0f}s")

        if budget < 5.0:
            print("Timer EMERGENCY: using deterministic pick (no LLM call).")
            self._consume_timer(battle, 0.5)
            return self._emergency_pick(battle)

        t0 = time.monotonic()
        try:
            decision_result = await asyncio.wait_for(
                self._get_llm_decision(battle_state_str),
                timeout=budget,
            )
        except asyncio.TimeoutError:
            print(f"Warning: LLM decision exceeded {budget:.0f}s budget. Choosing random action.")
            decision_result = {"error": f"timeout after {budget:.0f}s"}
        finally:
            self._consume_timer(battle, time.monotonic() - t0)
        print(decision_result)
        decision = decision_result.get("decision")
        error_message = decision_result.get("error")
        action_taken = False
        fallback_reason = ""

        if decision:
            function_name = decision.get("name")
            args = decision.get("arguments", {})
            reasoning = None
            if isinstance(args, dict):
                # Prefer the short, chat-ready summary; fall back to the first
                # sentence of the long analysis when the model skipped it.
                reasoning = args.get("reasoning")
                if not reasoning and args.get("analysis"):
                    reasoning = re.split(r"(?<=[.!?])\s+", args["analysis"].strip(), maxsplit=1)[0]
            analysis = args.get("analysis") if isinstance(args, dict) else None
            if function_name == "choose_move":
                move_name = args.get("move_name")
                if move_name:
                    chosen_move = self._find_move_by_name(battle, move_name)
                    if chosen_move and chosen_move in battle.available_moves:
                        action_taken = True
                        print(f"AI Decision: Using move '{chosen_move.id}'.")
                        self._record_turn(battle, f"used {chosen_move.id}", analysis, reasoning)
                        await self._send_reasoning(battle, reasoning)
                        return self.create_order(chosen_move)
                    else:
                        fallback_reason = f"LLM chose unavailable/invalid move '{move_name}'."
                else:
                    fallback_reason = "LLM 'choose_move' called without 'move_name'."
            elif function_name == "choose_switch":
                pokemon_name = args.get("pokemon_name")
                if pokemon_name:
                    chosen_switch = self._find_pokemon_by_name(battle, pokemon_name)
                    if chosen_switch and chosen_switch in battle.available_switches:
                        action_taken = True
                        print(f"AI Decision: Switching to '{chosen_switch.species}'.")
                        self._record_turn(
                            battle, f"switched to {chosen_switch.species}", analysis, reasoning
                        )
                        await self._send_reasoning(battle, reasoning)
                        return self.create_order(chosen_switch)
                    else:
                        fallback_reason = f"LLM chose unavailable/invalid switch '{pokemon_name}'."
                else:
                    fallback_reason = "LLM 'choose_switch' called without 'pokemon_name'."
            else:
                fallback_reason = f"LLM called unknown function '{function_name}'."

        if not action_taken:
            if not fallback_reason:
                if error_message:
                    fallback_reason = f"API Error: {error_message}"
                elif decision is None:
                    fallback_reason = "LLM did not provide a valid function call."
                else:
                    fallback_reason = "Unknown error processing LLM decision."

            print(f"Warning: {fallback_reason} Choosing random action.")

            if battle.available_moves or battle.available_switches:
                return self.choose_random_move(battle)
            else:
                print("AI Fallback: No moves or switches available. Using Struggle/Default.")
                return self.choose_default_move(battle)

    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement _get_llm_decision")


class GeminiAgent(LLMAgentBase):
    """Uses Google Gemini API for decisions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        avatar: str = "steven",
        *args,
        **kwargs,
    ):
        # Initialize LLM-side state BEFORE super().__init__(): poke-env's Player
        # constructor opens the websocket and may receive battle events that
        # invoke choose_move before this method returns.
        self.model_name = model
        used_api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not used_api_key:
            raise ValueError("Google API key not provided or found in GOOGLE_API_KEY env var.")
        self.genai_client = genai.Client(api_key=used_api_key)
        self.function_declarations = list(STANDARD_TOOL_SCHEMA.values())

        kwargs["avatar"] = avatar
        kwargs["start_timer_on_battle_start"] = True
        super().__init__(*args, **kwargs)

    async def _call_with_backoff(self, prompt: str, config, max_retries: int = 5):
        """Call generate_content with retries that respect Gemini's retryDelay on 429s."""
        for attempt in range(max_retries + 1):
            try:
                return await asyncio.to_thread(
                    self.genai_client.models.generate_content,
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
            except ClientError as e:
                if e.code != 429 or attempt == max_retries:
                    raise
                hint = self._extract_retry_delay(str(e))
                wait = max(hint or 0.0, 2 ** attempt)
                print(f"Gemini 429: backing off {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait)

    @staticmethod
    def _extract_retry_delay(message: str) -> Optional[float]:
        match = re.search(r"'retryDelay':\s*'([\d.]+)s'", message)
        return float(match.group(1)) if match else None

    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        prompt = (
            "You are a competitive Pokemon battle expert. For every turn, follow this two-step protocol:\n"
            "  1. ANALYZE first. Fill the required 'analysis' field with 3-5 sentences covering type matchups, "
            "speed, HP/status, threats, and which option (move or switch) is best AND WHY vs the alternatives.\n"
            "  2. THEN call the action. Set move_name or pokemon_name to the chosen action — must be from the "
            "available list, exact ID or species name.\n"
            "Also fill 'reasoning' with one short sentence (<250 chars) for the in-game chat.\n"
            "The state below already includes pre-computed type effectiveness ('Eff: 2x' on each move) and a "
            "speed comparison — use them, don't re-derive them.\n\n"
            f"Current Battle State:\n{battle_state}\n\n"
            "Choose the best action by calling 'choose_move' or 'choose_switch'."
        )

        try:
            tools = types.Tool(function_declarations=self.function_declarations)
            config = types.GenerateContentConfig(
                tools=[tools],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )

            response = await self._call_with_backoff(prompt, config)
            function_calls = response.function_calls
            if function_calls:
                return {
                    "decision": {
                        "name": function_calls[0].name,
                        "arguments": dict(function_calls[0].args),
                    }
                }
            return {"error": "Gemini did not return a function call."}

        except Exception as e:
            print(f"Unexpected error during Gemini processing: {e}")
            import traceback

            traceback.print_exc()
            return {"error": f"Unexpected error: {str(e)}"}


# OpenAI-style tool schema (LM Studio speaks the OpenAI chat-completions API).
_OPENAI_TOOLS = [
    {"type": "function", "function": spec} for spec in STANDARD_TOOL_SCHEMA.values()
]


class LMStudioAgent(LLMAgentBase):
    """Uses a local LM Studio server (OpenAI-compatible) for decisions."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",  # LM Studio ignores the value but the SDK requires one
        model: Optional[str] = None,  # None = whatever model is currently loaded
        avatar: str = "red",
        *args,
        **kwargs,
    ):
        # See GeminiAgent: assign before super().__init__() to avoid an init race.
        self.model_name = model
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.tools = _OPENAI_TOOLS

        kwargs["avatar"] = avatar
        kwargs["start_timer_on_battle_start"] = True
        super().__init__(*args, **kwargs)

    async def _resolve_model(self) -> str:
        if self.model_name:
            return self.model_name
        models = await self.client.models.list()
        if not models.data:
            raise RuntimeError("No model loaded in LM Studio. Load one in the UI first.")
        self.model_name = models.data[0].id
        print(f"LM Studio: using loaded model '{self.model_name}'")
        return self.model_name

    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a competitive Pokemon battle expert. Goal: win.\n"
            "For every turn, follow this two-step protocol:\n"
            "  1. ANALYZE first. Fill the required 'analysis' field with 3-5 sentences covering type matchups, "
            "speed, HP/status, threats, and why your choice beats the alternatives.\n"
            "  2. THEN pick the action: set move_name or pokemon_name to a value from the available list "
            "(exact move ID or species name).\n"
            "Also fill 'reasoning' with one short sentence (<250 chars) for the in-game chat.\n"
            "The battle state already has pre-computed type effectiveness ('Eff: 2x' on each move) and a "
            "speed comparison — use them, don't re-derive them."
        )
        user_prompt = (
            f"Current Battle State:\n{battle_state}\n\n"
            "Choose the best action by calling the appropriate function ('choose_move' or 'choose_switch')."
        )

        try:
            model = await self._resolve_model()
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=self.tools,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=1024,
            )
            message = response.choices[0].message
            if not message.tool_calls:
                return {"error": f"LM Studio model did not call a tool. Said: {message.content!r}"}

            call = message.tool_calls[0]
            function_name = call.function.name
            if function_name not in STANDARD_TOOL_SCHEMA:
                return {"error": f"Model called unknown function '{function_name}'."}
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                return {"error": f"Could not decode tool arguments: {call.function.arguments!r}"}
            return {"decision": {"name": function_name, "arguments": arguments}}

        except Exception as e:
            print(f"Unexpected error during LM Studio call: {e}")
            return {"error": f"Unexpected error: {e}"}
