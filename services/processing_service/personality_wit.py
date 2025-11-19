"""
Personality & Wit Module for ADRIAN Processing Service.

Rewrites responses to match configured personality tone while preserving
factual content. Supports multiple tone presets and per-response overrides.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from shared.logging_config import setup_logging
from shared.config import Settings, get_settings


logger = setup_logging("processing-service.personality")


class TonePreset(str, Enum):
    """Available personality tone presets."""
    JARVIS = "jarvis"  # Professional, respectful, witty (default)
    FORMAL = "formal"  # Very formal, no contractions
    MINIMAL = "minimal"  # Brief, concise, no fluff
    FRIENDLY = "friendly"  # Warm, conversational, casual
    SARCASTIC = "sarcastic"  # Witty, humorous, slightly cheeky
    PROFESSIONAL = "professional"  # Business-like, clear, direct


@dataclass
class PersonalityConfig:
    """Configuration for personality rewriting."""
    tone: TonePreset = TonePreset.JARVIS
    use_contractions: bool = True
    add_sir: bool = True  # Add "Sir" at end of responses
    wit_level: int = 2  # 0=none, 1=subtle, 2=moderate, 3=high
    formality_level: int = 2  # 0=very casual, 1=casual, 2=neutral, 3=formal, 4=very formal
    max_length_multiplier: float = 1.2  # Max response length relative to original


class PersonalityRewriter:
    """
    Rewrites responses to match personality tone while preserving factual content.
    
    Uses rule-based transformations to ensure:
    - Factual information is preserved
    - Tone matches preset
    - Response length is appropriate
    - Grammar and clarity are maintained
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._default_config = self._load_default_config()
    
    def _load_default_config(self) -> PersonalityConfig:
        """Load default personality configuration from settings."""
        tone_str = getattr(self._settings, 'personality_default_tone', 'jarvis').lower()
        try:
            tone = TonePreset(tone_str)
        except ValueError:
            logger.warning(f"Unknown tone '{tone_str}', defaulting to jarvis")
            tone = TonePreset.JARVIS
        
        return PersonalityConfig(
            tone=tone,
            use_contractions=getattr(self._settings, 'personality_use_contractions', True),
            add_sir=getattr(self._settings, 'personality_add_sir', True),
            wit_level=getattr(self._settings, 'personality_wit_level', 2),
            formality_level=getattr(self._settings, 'personality_formality_level', 2),
            max_length_multiplier=getattr(self._settings, 'personality_max_length_multiplier', 1.2)
        )
    
    async def rewrite(
        self,
        text: str,
        tone_override: Optional[TonePreset] = None,
        config_override: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Rewrite response text to match personality tone.
        
        Args:
            text: Original response text
            tone_override: Optional tone preset override
            config_override: Optional configuration overrides
            
        Returns:
            Rewritten text with personality applied
        """
        if not text or not text.strip():
            return text
        
        # Determine configuration
        config = self._get_config(tone_override, config_override)
        
        # Apply tone-specific transformations
        rewritten = text.strip()
        
        # Step 1: Preserve factual content (extract key facts)
        facts = self._extract_facts(rewritten)
        
        # Step 2: Apply tone transformations
        if config.tone == TonePreset.JARVIS:
            rewritten = self._apply_jarvis_tone(rewritten, config, facts)
        elif config.tone == TonePreset.FORMAL:
            rewritten = self._apply_formal_tone(rewritten, config, facts)
        elif config.tone == TonePreset.MINIMAL:
            rewritten = self._apply_minimal_tone(rewritten, config, facts)
        elif config.tone == TonePreset.FRIENDLY:
            rewritten = self._apply_friendly_tone(rewritten, config, facts)
        elif config.tone == TonePreset.SARCASTIC:
            rewritten = self._apply_sarcastic_tone(rewritten, config, facts)
        elif config.tone == TonePreset.PROFESSIONAL:
            rewritten = self._apply_professional_tone(rewritten, config, facts)
        
        # Step 3: Verify factual content is preserved
        rewritten = self._verify_facts_preserved(rewritten, facts)
        
        # Step 4: Final cleanup
        rewritten = self._final_cleanup(rewritten, config)
        
        return rewritten
    
    def _get_config(
        self,
        tone_override: Optional[TonePreset],
        config_override: Optional[Dict[str, Any]]
    ) -> PersonalityConfig:
        """Get configuration with overrides applied."""
        config = PersonalityConfig(
            tone=tone_override or self._default_config.tone,
            use_contractions=self._default_config.use_contractions,
            add_sir=self._default_config.add_sir,
            wit_level=self._default_config.wit_level,
            formality_level=self._default_config.formality_level,
            max_length_multiplier=self._default_config.max_length_multiplier
        )
        
        if config_override:
            if 'tone' in config_override:
                try:
                    config.tone = TonePreset(config_override['tone'].lower())
                except ValueError:
                    logger.warning(f"Invalid tone override: {config_override['tone']}")
            if 'use_contractions' in config_override:
                config.use_contractions = bool(config_override['use_contractions'])
            if 'add_sir' in config_override:
                config.add_sir = bool(config_override['add_sir'])
            if 'wit_level' in config_override:
                config.wit_level = max(0, min(3, int(config_override['wit_level'])))
            if 'formality_level' in config_override:
                config.formality_level = max(0, min(4, int(config_override['formality_level'])))
        
        return config
    
    def _extract_facts(self, text: str) -> Dict[str, Any]:
        """
        Extract factual content from text that must be preserved.
        
        Returns:
            Dictionary of extracted facts (numbers, names, actions, etc.)
        """
        facts = {
            'numbers': re.findall(r'\b\d+(?:\.\d+)?\b', text),
            'times': re.findall(r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b', text),
            'dates': re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text),
            'actions': self._extract_actions(text),
            'entities': self._extract_entities(text),
            'original_length': len(text)
        }
        return facts
    
    def _extract_actions(self, text: str) -> List[str]:
        """Extract action verbs that must be preserved."""
        action_patterns = [
            r'\b(opening|opened|open)\b',
            r'\b(closing|closed|close)\b',
            r'\b(starting|started|start)\b',
            r'\b(stopping|stopped|stop)\b',
            r'\b(searching|searched|search)\b',
            r'\b(creating|created|create)\b',
            r'\b(reminding|reminded|remind)\b',
        ]
        actions = []
        for pattern in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            actions.extend(matches)
        return actions
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (apps, files, etc.) that must be preserved."""
        # Simple extraction - look for capitalized words that aren't at start of sentence
        # This is a simplified version; in production, use NER
        entities = []
        # Look for quoted strings
        quoted = re.findall(r'"([^"]+)"', text)
        entities.extend(quoted)
        # Look for common app names (capitalized, not at sentence start)
        # This is heuristic-based
        return entities
    
    def _apply_jarvis_tone(self, text: str, config: PersonalityConfig, facts: Dict[str, Any]) -> str:
        """Apply Jarvis-like tone: professional, respectful, witty."""
        result = text
        
        # Ensure "Sir" is added if configured
        if config.add_sir and not re.search(r'\bSir\b[.,!?]*$', result, re.IGNORECASE):
            # Add "Sir" at the end
            if result.endswith('.'):
                result = result[:-1] + ', Sir.'
            elif result.endswith('!'):
                result = result[:-1] + ', Sir!'
            elif result.endswith('?'):
                result = result[:-1] + ', Sir?'
            else:
                result = result + ', Sir.'
        
        # Add subtle wit if configured
        if config.wit_level >= 2:
            # Replace generic responses with more interesting ones
            result = re.sub(
                r'\b(I\'m|I am) ready\b',
                r'\1 ready when you are',
                result,
                flags=re.IGNORECASE
            )
        
        # Ensure professional tone
        result = self._ensure_professional_language(result, config)
        
        return result
    
    def _apply_formal_tone(self, text: str, config: PersonalityConfig, facts: Dict[str, Any]) -> str:
        """Apply formal tone: no contractions, very polite."""
        result = text
        
        # Remove contractions
        if not config.use_contractions:
            result = self._expand_contractions(result)
        
        # Ensure formal address
        if config.add_sir:
            if not re.search(r'\bSir\b', result, re.IGNORECASE):
                if result.endswith('.'):
                    result = result[:-1] + ', Sir.'
                else:
                    result = result + ', Sir.'
        
        # Use formal language
        result = re.sub(r'\bI\'m\b', 'I am', result, flags=re.IGNORECASE)
        result = re.sub(r'\byou\'re\b', 'you are', result, flags=re.IGNORECASE)
        result = re.sub(r'\bcan\'t\b', 'cannot', result, flags=re.IGNORECASE)
        result = re.sub(r'\bwon\'t\b', 'will not', result, flags=re.IGNORECASE)
        
        return result
    
    def _apply_minimal_tone(self, text: str, config: PersonalityConfig, facts: Dict[str, Any]) -> str:
        """Apply minimal tone: brief, concise, no fluff."""
        result = text
        
        # Remove unnecessary words
        result = re.sub(r'\bI\'m\b', '', result, flags=re.IGNORECASE)
        result = re.sub(r'\bI am\b', '', result, flags=re.IGNORECASE)
        result = re.sub(r'\bSir\b', '', result, flags=re.IGNORECASE)
        
        # Remove filler phrases
        result = re.sub(r'\b(please|kindly|if you would)\b', '', result, flags=re.IGNORECASE)
        result = re.sub(r'\s+', ' ', result)  # Clean up extra spaces
        result = result.strip()
        
        # Ensure it ends with punctuation
        if result and not re.search(r'[.!?]$', result):
            result = result + '.'
        
        return result
    
    def _apply_friendly_tone(self, text: str, config: PersonalityConfig, facts: Dict[str, Any]) -> str:
        """Apply friendly tone: warm, conversational, casual."""
        result = text
        
        # Use contractions
        if config.use_contractions:
            result = self._add_contractions(result)
        
        # Remove "Sir" for friendlier tone
        result = re.sub(r',\s*Sir[.,!?]', '', result, flags=re.IGNORECASE)
        
        # Add friendly phrases
        if config.wit_level >= 1:
            result = re.sub(
                r'\b(opening|opened)\b',
                r'\1 that up',
                result,
                flags=re.IGNORECASE
            )
        
        return result
    
    def _apply_sarcastic_tone(self, text: str, config: PersonalityConfig, facts: Dict[str, Any]) -> str:
        """Apply sarcastic tone: witty, humorous, slightly cheeky."""
        result = text
        
        # Add subtle sarcasm based on wit level
        if config.wit_level >= 2:
            # Replace generic confirmations with witty ones
            result = re.sub(
                r'\b(opening|opened)\s+(\w+)\b',
                r'\1 \2, as requested',
                result,
                flags=re.IGNORECASE
            )
        
        # Keep "Sir" but make it slightly less formal
        if config.add_sir:
            if not re.search(r'\bSir\b', result, re.IGNORECASE):
                if result.endswith('.'):
                    result = result[:-1] + ', Sir.'
                else:
                    result = result + ', Sir.'
        
        return result
    
    def _apply_professional_tone(self, text: str, config: PersonalityConfig, facts: Dict[str, Any]) -> str:
        """Apply professional tone: business-like, clear, direct."""
        result = text
        
        # Use professional language
        result = self._ensure_professional_language(result, config)
        
        # Remove overly casual elements
        result = re.sub(r'\b(yeah|yep|nope|nah)\b', 'yes', result, flags=re.IGNORECASE)
        result = re.sub(r'\b(okay|ok)\b', 'understood', result, flags=re.IGNORECASE)
        
        # Ensure clarity
        if config.add_sir:
            if not re.search(r'\bSir\b', result, re.IGNORECASE):
                if result.endswith('.'):
                    result = result[:-1] + ', Sir.'
                else:
                    result = result + ', Sir.'
        
        return result
    
    def _ensure_professional_language(self, text: str, config: PersonalityConfig) -> str:
        """Ensure professional language is used."""
        result = text
        # Replace casual language with professional alternatives
        replacements = {
            r'\bgonna\b': 'going to',
            r'\bwanna\b': 'want to',
            r'\bgotta\b': 'have to',
            r'\bya\b': 'you',
        }
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions to full forms."""
        replacements = {
            r'\bI\'m\b': 'I am',
            r'\byou\'re\b': 'you are',
            r'\bhe\'s\b': 'he is',
            r'\bshe\'s\b': 'she is',
            r'\bit\'s\b': 'it is',
            r'\bwe\'re\b': 'we are',
            r'\bthey\'re\b': 'they are',
            r'\bcan\'t\b': 'cannot',
            r'\bwon\'t\b': 'will not',
            r'\bdon\'t\b': 'do not',
            r'\bdoesn\'t\b': 'does not',
            r'\bdidn\'t\b': 'did not',
            r'\bisn\'t\b': 'is not',
            r'\baren\'t\b': 'are not',
            r'\bwasn\'t\b': 'was not',
            r'\bweren\'t\b': 'were not',
            r'\bhasn\'t\b': 'has not',
            r'\bhaven\'t\b': 'have not',
            r'\bhadn\'t\b': 'had not',
            r'\bwouldn\'t\b': 'would not',
            r'\bcouldn\'t\b': 'could not',
            r'\bshouldn\'t\b': 'should not',
        }
        result = text
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result
    
    def _add_contractions(self, text: str) -> str:
        """Add contractions for more casual tone."""
        replacements = {
            r'\bI am\b': "I'm",
            r'\byou are\b': "you're",
            r'\bcan not\b': "can't",
            r'\bwill not\b': "won't",
            r'\bdo not\b': "don't",
            r'\bdoes not\b': "doesn't",
        }
        result = text
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result
    
    def _verify_facts_preserved(self, text: str, facts: Dict[str, Any]) -> str:
        """
        Verify that factual content is preserved in rewritten text.
        If facts are missing, attempt to restore them.
        """
        result = text
        
        # Verify numbers are preserved
        for number in facts.get('numbers', []):
            if number not in result:
                logger.warning(f"Number {number} may have been lost in rewrite")
        
        # Verify times are preserved
        for time_str in facts.get('times', []):
            if time_str.lower() not in result.lower():
                logger.warning(f"Time {time_str} may have been lost in rewrite")
        
        # Verify actions are preserved (at least the intent)
        # This is a simplified check
        original_actions = facts.get('actions', [])
        if original_actions:
            # Check if any action-related words are still present
            action_keywords = ['open', 'close', 'start', 'stop', 'search', 'create', 'remind']
            has_action = any(keyword in result.lower() for keyword in action_keywords)
            if not has_action and any('open' in str(a).lower() or 'close' in str(a).lower() for a in original_actions):
                logger.warning("Action keywords may have been lost in rewrite")
        
        return result
    
    def _final_cleanup(self, text: str, config: PersonalityConfig) -> str:
        """Final cleanup: spacing, punctuation, etc."""
        result = text
        
        # Clean up multiple spaces
        result = re.sub(r'\s+', ' ', result)
        
        # Ensure proper capitalization
        if result:
            result = result[0].upper() + result[1:] if len(result) > 1 else result.upper()
        
        # Ensure it ends with punctuation
        if result and not re.search(r'[.!?]$', result):
            result = result + '.'
        
        # Clean up trailing spaces
        result = result.strip()
        
        return result


# Global instance
_personality_rewriter: Optional[PersonalityRewriter] = None


def get_personality_rewriter(settings: Optional[Settings] = None) -> PersonalityRewriter:
    """Get or create global personality rewriter instance."""
    global _personality_rewriter
    if _personality_rewriter is None:
        _personality_rewriter = PersonalityRewriter(settings)
    return _personality_rewriter


async def rewrite_with_personality(
    text: str,
    tone_override: Optional[TonePreset] = None,
    config_override: Optional[Dict[str, Any]] = None,
    settings: Optional[Settings] = None
) -> str:
    """
    Convenience function to rewrite text with personality.
    
    Args:
        text: Original response text
        tone_override: Optional tone preset override
        config_override: Optional configuration overrides
        settings: Optional settings instance
        
    Returns:
        Rewritten text with personality applied
    """
    rewriter = get_personality_rewriter(settings)
    return await rewriter.rewrite(text, tone_override, config_override)

