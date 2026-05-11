"""LLM-driven Pokémon battle agents.

Adapted from the Hugging Face Agents Course (Bonus Unit 3) reference at
https://huggingface.co/spaces/Jofthomas/twitch_streaming/blob/main/agents.py
"""
import asyncio
import json
import os
import re
from typing import Any, Dict, Optional

# --- Google Gemini ---
from google import genai
from google.genai import types
from google.genai.errors import ClientError

# --- OpenAI-compatible (LM Studio) ---
from openai import AsyncOpenAI

# --- Poke-Env ---
from poke_env.player import Player


def normalize_name(name: str) -> str:
    """Lowercase and remove non-alphanumeric characters."""
    return "".join(filter(str.isalnum, name)).lower()


_REASONING_PROP = {
    "reasoning": {
        "type": "string",
        "description": "One short sentence (under 250 chars) explaining why this action was chosen. Will be sent as a chat message in the battle room.",
    }
}

STANDARD_TOOL_SCHEMA = {
    "choose_move": {
        "name": "choose_move",
        "description": "Selects and executes an available attacking or status move.",
        "parameters": {
            "type": "object",
            "properties": {
                "move_name": {
                    "type": "string",
                    "description": "The exact name or ID (e.g., 'thunderbolt', 'swordsdance') of the move to use. Must be one of the available moves.",
                },
                **_REASONING_PROP,
            },
            "required": ["move_name"],
        },
    },
    "choose_switch": {
        "name": "choose_switch",
        "description": "Selects an available Pokémon from the bench to switch into.",
        "parameters": {
            "type": "object",
            "properties": {
                "pokemon_name": {
                    "type": "string",
                    "description": "The exact name of the Pokémon species to switch to (e.g., 'Pikachu', 'Charizard'). Must be one of the available switches.",
                },
                **_REASONING_PROP,
            },
            "required": ["pokemon_name"],
        },
    },
}

CHAT_MAX_LEN = 280


class LLMAgentBase(Player):
    # Hard cap on how long we wait for a single LLM decision. Showdown's per-turn
    # budget is ~150s baseline; staying well under that prevents losing on time.
    decision_timeout: float = 90.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.standard_tools = STANDARD_TOOL_SCHEMA
        self.battle_history = []

    def _format_battle_state(self, battle) -> str:
        active_pkmn = battle.active_pokemon
        active_pkmn_info = (
            f"Your active Pokemon: {active_pkmn.species} "
            f"(Type: {'/'.join(map(str, active_pkmn.types))}) "
            f"HP: {active_pkmn.current_hp_fraction * 100:.1f}% "
            f"Status: {active_pkmn.status.name if active_pkmn.status else 'None'} "
            f"Boosts: {active_pkmn.boosts}"
        )

        opponent_pkmn = battle.opponent_active_pokemon
        opp_info_str = "Unknown"
        if opponent_pkmn:
            opp_info_str = (
                f"{opponent_pkmn.species} "
                f"(Type: {'/'.join(map(str, opponent_pkmn.types))}) "
                f"HP: {opponent_pkmn.current_hp_fraction * 100:.1f}% "
                f"Status: {opponent_pkmn.status.name if opponent_pkmn.status else 'None'} "
                f"Boosts: {opponent_pkmn.boosts}"
            )
        opponent_pkmn_info = f"Opponent's active Pokemon: {opp_info_str}"

        available_moves_info = "Available moves:\n"
        if battle.available_moves:
            available_moves_info += "\n".join(
                [
                    f"- {move.id} (Type: {move.type}, BP: {move.base_power}, Acc: {move.accuracy}, PP: {move.current_pp}/{move.max_pp}, Cat: {move.category.name})"
                    for move in battle.available_moves
                ]
            )
        else:
            available_moves_info += "- None (Must switch or Struggle)"

        available_switches_info = "Available switches:\n"
        if battle.available_switches:
            available_switches_info += "\n".join(
                [
                    f"- {pkmn.species} (HP: {pkmn.current_hp_fraction * 100:.1f}%, Status: {pkmn.status.name if pkmn.status else 'None'})"
                    for pkmn in battle.available_switches
                ]
            )
        else:
            available_switches_info += "- None"

        state_str = (
            f"{active_pkmn_info}\n"
            f"{opponent_pkmn_info}\n\n"
            f"{available_moves_info}\n\n"
            f"{available_switches_info}\n\n"
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
        for move in battle.available_moves:
            if move.name.lower() == move_name.lower():
                print(
                    f"Warning: Matched move by display name '{move.name}' instead of ID '{move.id}'. Input was '{move_name}'."
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

    async def choose_move(self, battle) -> str:
        battle_state_str = self._format_battle_state(battle)
        try:
            decision_result = await asyncio.wait_for(
                self._get_llm_decision(battle_state_str),
                timeout=self.decision_timeout,
            )
        except asyncio.TimeoutError:
            print(
                f"Warning: LLM decision exceeded {self.decision_timeout}s timeout. Choosing random action."
            )
            decision_result = {"error": f"timeout after {self.decision_timeout}s"}
        print(decision_result)
        decision = decision_result.get("decision")
        error_message = decision_result.get("error")
        action_taken = False
        fallback_reason = ""

        if decision:
            function_name = decision.get("name")
            args = decision.get("arguments", {})
            reasoning = args.get("reasoning") if isinstance(args, dict) else None
            if function_name == "choose_move":
                move_name = args.get("move_name")
                if move_name:
                    chosen_move = self._find_move_by_name(battle, move_name)
                    if chosen_move and chosen_move in battle.available_moves:
                        action_taken = True
                        print(f"AI Decision: Using move '{chosen_move.id}'.")
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
            "Based on the current battle state, decide the best action: either use an available move or switch to an available Pokémon. "
            "Consider type matchups, HP, status conditions, field effects, entry hazards, and potential opponent actions. "
            "Only choose actions listed as available using their exact ID (for moves) or species name (for switches). "
            "Use the provided functions to indicate your choice. Always include a 'reasoning' field with one short sentence "
            "(under 250 chars) explaining your choice — it will be sent as a chat message in the battle room.\n\n"
            f"Current Battle State:\n{battle_state}\n\n"
            "Choose the best action by calling the appropriate function ('choose_move' or 'choose_switch')."
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
            "You are a skilled Pokemon battle AI. Your goal is to win the battle. "
            "Based on the current battle state, decide the best action: either use an available move or switch to an available Pokémon. "
            "Consider type matchups, HP, status conditions, field effects, entry hazards, and potential opponent actions. "
            "Only choose actions listed as available using their exact ID (for moves) or species name (for switches). "
            "Use the provided tools to indicate your choice. Always include a 'reasoning' field with one short sentence "
            "(under 250 chars) explaining your choice — it will be sent as a chat message in the battle room."
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
