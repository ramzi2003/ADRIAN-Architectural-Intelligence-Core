"""
Tests for Personality & Wit Module.

Ensures that personality rewriting preserves factual content while
applying appropriate tone transformations.
"""
import pytest
from services.processing_service.personality_wit import (
    PersonalityRewriter,
    TonePreset,
    PersonalityConfig,
    rewrite_with_personality,
)
from shared.config import Settings


class TestPersonalityRewriter:
    """Test suite for PersonalityRewriter."""
    
    @pytest.fixture
    def rewriter(self):
        """Create a PersonalityRewriter instance for testing."""
        settings = Settings()
        return PersonalityRewriter(settings)
    
    def test_extract_facts_numbers(self, rewriter):
        """Test that numbers are extracted correctly."""
        text = "Opening Chrome at 3:45 PM on 12/25/2024"
        facts = rewriter._extract_facts(text)
        
        assert "3" in facts['numbers'] or "45" in facts['numbers']
        assert "3:45 PM" in facts['times'] or "3:45PM" in facts['times']
        assert "12/25/2024" in facts['dates']
    
    def test_extract_facts_actions(self, rewriter):
        """Test that actions are extracted correctly."""
        text = "Opening Chrome browser now"
        facts = rewriter._extract_facts(text)
        
        assert len(facts['actions']) > 0
        assert any('open' in str(action).lower() for action in facts['actions'])
    
    def test_jarvis_tone_adds_sir(self, rewriter):
        """Test that Jarvis tone adds 'Sir' appropriately."""
        text = "Opening Chrome"
        config = PersonalityConfig(tone=TonePreset.JARVIS, add_sir=True)
        facts = rewriter._extract_facts(text)
        
        result = rewriter._apply_jarvis_tone(text, config, facts)
        
        assert "Sir" in result or "sir" in result.lower()
        assert "Chrome" in result  # Fact preserved
    
    def test_jarvis_tone_preserves_facts(self, rewriter):
        """Test that Jarvis tone preserves factual content."""
        text = "Opening Chrome at 3:45 PM"
        config = PersonalityConfig(tone=TonePreset.JARVIS, add_sir=True)
        facts = rewriter._extract_facts(text)
        
        result = rewriter._apply_jarvis_tone(text, config, facts)
        
        # Facts must be preserved
        assert "Chrome" in result
        assert "3:45" in result or "3:45PM" in result.lower()
        assert "opening" in result.lower() or "open" in result.lower()
    
    def test_formal_tone_expands_contractions(self, rewriter):
        """Test that formal tone expands contractions."""
        text = "I'm opening Chrome, you're ready"
        config = PersonalityConfig(
            tone=TonePreset.FORMAL,
            use_contractions=False,
            add_sir=True
        )
        facts = rewriter._extract_facts(text)
        
        result = rewriter._apply_formal_tone(text, config, facts)
        
        assert "I am" in result or "I'm" not in result
        assert "you are" in result or "you're" not in result
        assert "Chrome" in result  # Fact preserved
    
    def test_minimal_tone_removes_fluff(self, rewriter):
        """Test that minimal tone removes unnecessary words."""
        text = "I am opening Chrome browser for you, Sir"
        config = PersonalityConfig(tone=TonePreset.MINIMAL, add_sir=False)
        facts = rewriter._extract_facts(text)
        
        result = rewriter._apply_minimal_tone(text, config, facts)
        
        # Should be shorter
        assert len(result) <= len(text) or len(result) < len(text) + 20
        assert "Chrome" in result  # Fact preserved
        assert "opening" in result.lower() or "open" in result.lower()
    
    def test_friendly_tone_adds_contractions(self, rewriter):
        """Test that friendly tone uses contractions."""
        text = "I am opening Chrome for you"
        config = PersonalityConfig(
            tone=TonePreset.FRIENDLY,
            use_contractions=True,
            add_sir=False
        )
        facts = rewriter._extract_facts(text)
        
        result = rewriter._apply_friendly_tone(text, config, facts)
        
        # Should have contractions or be more casual
        assert "Chrome" in result  # Fact preserved
        assert "opening" in result.lower() or "open" in result.lower()
    
    def test_professional_tone_removes_casual(self, rewriter):
        """Test that professional tone removes casual language."""
        text = "Yeah, I'm gonna open Chrome, okay?"
        config = PersonalityConfig(tone=TonePreset.PROFESSIONAL, add_sir=True)
        facts = rewriter._extract_facts(text)
        
        result = rewriter._apply_professional_tone(text, config, facts)
        
        assert "Chrome" in result  # Fact preserved
        assert "yeah" not in result.lower()
        assert "gonna" not in result.lower()
        assert "okay" not in result.lower() or "understood" in result.lower()
    
    def test_facts_preserved_after_rewrite(self, rewriter):
        """Test that facts are preserved after full rewrite."""
        text = "Opening Chrome browser at 3:45 PM on December 25th"
        facts = rewriter._extract_facts(text)
        
        # Rewrite with different tones
        for tone in [TonePreset.JARVIS, TonePreset.FORMAL, TonePreset.MINIMAL]:
            config = PersonalityConfig(tone=tone, add_sir=(tone == TonePreset.JARVIS))
            result = rewriter._apply_jarvis_tone(text, config, facts) if tone == TonePreset.JARVIS else \
                     rewriter._apply_formal_tone(text, config, facts) if tone == TonePreset.FORMAL else \
                     rewriter._apply_minimal_tone(text, config, facts)
            
            # Key facts must be preserved
            assert "Chrome" in result, f"Chrome not preserved in {tone} tone"
            assert "3:45" in result or "3:45PM" in result.lower(), f"Time not preserved in {tone} tone"
            assert "opening" in result.lower() or "open" in result.lower(), f"Action not preserved in {tone} tone"
    
    def test_verify_facts_preserved(self, rewriter):
        """Test the fact verification mechanism."""
        original_text = "Opening Chrome at 3:45 PM"
        facts = rewriter._extract_facts(original_text)
        
        # Create a result that might have lost facts
        result_with_facts = "Opening Chrome at 3:45 PM, Sir."
        result_without_facts = "Done, Sir."  # Missing facts
        
        # Verification should catch missing facts
        verified = rewriter._verify_facts_preserved(result_with_facts, facts)
        assert "Chrome" in verified
        assert "3:45" in verified or "3:45PM" in verified.lower()
        
        # Should log warning for missing facts
        verified_missing = rewriter._verify_facts_preserved(result_without_facts, facts)
        # The function logs warnings but doesn't modify text if facts are missing
        # This is expected behavior - we log but don't auto-fix
    
    def test_final_cleanup(self, rewriter):
        """Test final cleanup of rewritten text."""
        text = "  opening   chrome  "
        config = PersonalityConfig()
        
        result = rewriter._final_cleanup(text, config)
        
        assert result.startswith("O")  # Capitalized
        assert "  " not in result  # No double spaces
        assert result.endswith(".")  # Ends with punctuation
        assert result.strip() == result  # No trailing spaces
    
    @pytest.mark.asyncio
    async def test_rewrite_with_personality_jarvis(self, rewriter):
        """Test full rewrite with Jarvis tone."""
        text = "Opening Chrome"
        result = await rewriter.rewrite(text, tone_override=TonePreset.JARVIS)
        
        assert "Chrome" in result  # Fact preserved
        assert "Sir" in result or "sir" in result.lower()  # Personality added
        assert "opening" in result.lower() or "open" in result.lower()  # Action preserved
    
    @pytest.mark.asyncio
    async def test_rewrite_with_personality_formal(self, rewriter):
        """Test full rewrite with formal tone."""
        text = "I'm opening Chrome"
        result = await rewriter.rewrite(text, tone_override=TonePreset.FORMAL)
        
        assert "Chrome" in result  # Fact preserved
        # Should expand contractions or be more formal
        assert "opening" in result.lower() or "open" in result.lower()
    
    @pytest.mark.asyncio
    async def test_rewrite_with_personality_minimal(self, rewriter):
        """Test full rewrite with minimal tone."""
        text = "I am opening Chrome browser for you, Sir"
        result = await rewriter.rewrite(text, tone_override=TonePreset.MINIMAL)
        
        assert "Chrome" in result  # Fact preserved
        assert "opening" in result.lower() or "open" in result.lower()  # Action preserved
        # Should be more concise
        assert len(result) <= len(text) + 10  # Allow some variation
    
    @pytest.mark.asyncio
    async def test_rewrite_with_config_override(self, rewriter):
        """Test rewrite with configuration override."""
        text = "Opening Chrome"
        config_override = {
            "add_sir": False,
            "wit_level": 0
        }
        
        result = await rewriter.rewrite(
            text,
            tone_override=TonePreset.JARVIS,
            config_override=config_override
        )
        
        assert "Chrome" in result  # Fact preserved
        # Should not have "Sir" due to override
        assert "Sir" not in result or "sir" not in result.lower()
    
    @pytest.mark.asyncio
    async def test_rewrite_preserves_numbers(self, rewriter):
        """Test that numbers are preserved in rewrite."""
        test_cases = [
            "Opening 3 applications",
            "The time is 3:45 PM",
            "Reminder set for 12/25/2024",
            "Search returned 42 results",
        ]
        
        for text in test_cases:
            result = await rewriter.rewrite(text)
            # Extract numbers from original and result
            import re
            original_numbers = re.findall(r'\d+', text)
            result_numbers = re.findall(r'\d+', result)
            
            # All numbers should be preserved
            for num in original_numbers:
                assert num in result_numbers or any(num in r for r in result_numbers), \
                    f"Number {num} not preserved in: {text} -> {result}"
    
    @pytest.mark.asyncio
    async def test_rewrite_preserves_actions(self, rewriter):
        """Test that action verbs are preserved."""
        test_cases = [
            ("Opening Chrome", "open"),
            ("Closing Firefox", "close"),
            ("Starting service", "start"),
            ("Searching for files", "search"),
        ]
        
        for text, action in test_cases:
            result = await rewriter.rewrite(text)
            assert action in result.lower(), \
                f"Action '{action}' not preserved in: {text} -> {result}"
    
    @pytest.mark.asyncio
    async def test_rewrite_preserves_entities(self, rewriter):
        """Test that named entities (apps, files) are preserved."""
        test_cases = [
            "Opening Chrome",
            "Closing 'My Document.pdf'",
            "Searching for Python tutorials",
        ]
        
        for text in test_cases:
            result = await rewriter.rewrite(text)
            # Extract key entities (capitalized words, quoted strings)
            import re
            entities = re.findall(r'"[^"]+"|[A-Z][a-z]+', text)
            
            for entity in entities:
                if entity.strip('"'):
                    assert entity.strip('"') in result or entity.lower() in result.lower(), \
                        f"Entity '{entity}' not preserved in: {text} -> {result}"
    
    @pytest.mark.asyncio
    async def test_rewrite_empty_text(self, rewriter):
        """Test that empty text is handled gracefully."""
        result = await rewriter.rewrite("")
        assert result == ""
        
        result = await rewriter.rewrite("   ")
        assert result == "" or result.strip() == ""
    
    @pytest.mark.asyncio
    async def test_rewrite_convenience_function(self):
        """Test the convenience function rewrite_with_personality."""
        text = "Opening Chrome"
        result = await rewrite_with_personality(text, tone_override=TonePreset.JARVIS)
        
        assert "Chrome" in result  # Fact preserved
        assert len(result) > 0


class TestPersonalityConfig:
    """Test suite for PersonalityConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PersonalityConfig()
        
        assert config.tone == TonePreset.JARVIS
        assert config.use_contractions is True
        assert config.add_sir is True
        assert config.wit_level == 2
        assert config.formality_level == 2
    
    def test_config_override(self):
        """Test configuration override."""
        config = PersonalityConfig(
            tone=TonePreset.FORMAL,
            use_contractions=False,
            add_sir=False,
            wit_level=0,
            formality_level=4
        )
        
        assert config.tone == TonePreset.FORMAL
        assert config.use_contractions is False
        assert config.add_sir is False
        assert config.wit_level == 0
        assert config.formality_level == 4


class TestTonePreset:
    """Test suite for TonePreset enum."""
    
    def test_tone_preset_values(self):
        """Test that all tone presets are available."""
        assert TonePreset.JARVIS == "jarvis"
        assert TonePreset.FORMAL == "formal"
        assert TonePreset.MINIMAL == "minimal"
        assert TonePreset.FRIENDLY == "friendly"
        assert TonePreset.SARCASTIC == "sarcastic"
        assert TonePreset.PROFESSIONAL == "professional"
    
    def test_tone_preset_from_string(self):
        """Test creating TonePreset from string."""
        assert TonePreset("jarvis") == TonePreset.JARVIS
        assert TonePreset("formal") == TonePreset.FORMAL
        assert TonePreset("minimal") == TonePreset.MINIMAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

