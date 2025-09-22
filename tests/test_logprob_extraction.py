#!/usr/bin/env python3
"""
Test script to verify logprob extraction for judge outputs.
"""

import re
import sys
import yaml
import json
from pprint import pprint

class MockLogProbs:
    """Mock class to simulate logprobs object"""
    def __init__(self, tokens_with_probs):
        self.content = []
        for text, logprob in tokens_with_probs:
            self.content.append(MockToken(text, logprob))

class MockToken:
    """Mock class to simulate token object in logprobs"""
    def __init__(self, text, logprob):
        self.text = text
        self.logprob = logprob

def calculate_token_logprobs(judgment, match, logprobs):
    """Helper function to calculate token logprobs for a match, from gen_judgment.py"""
    token_logprobs = []
    
    # Find the match in the full text
    match_start = judgment.find(match)
    if match_start >= 0:
        match_end = match_start + len(match)
        
        # Go through tokens and find those within our match
        token_offset = 0
        for i, token in enumerate(logprobs.content):
            if hasattr(token, "text") and hasattr(token, "logprob"):
                # Track the token offset in the original text
                token_text = token.text
                token_length = len(token_text)
                
                # Check if this token is within our match
                token_start = token_offset
                token_end = token_start + token_length
                
                # If there's overlap with our match, include this token's logprob
                if (token_start >= match_start and token_start < match_end) or \
                   (token_end > match_start and token_end <= match_end) or \
                   (token_start <= match_start and token_end >= match_end):
                    token_logprobs.append(token.logprob)
                
                token_offset += token_length
    
    return token_logprobs

def get_score_logprobs(judgment, pattern, logprobs):
    """Simplified version of get_score_logprobs from gen_judgment.py"""
    matches = pattern.findall(judgment)
    matches = [m for m in matches if m != ""]
    
    if len(set(matches)) == 0:
        return None, True  # No matches
    elif len(set(matches)) == 1:
        match = matches[0].strip("\n")
        
        # Calculate average logprob for the matched tokens
        avg_logprob = None
        
        # Only process if we have valid logprobs data
        if logprobs and hasattr(logprobs, "content"):
            token_logprobs = calculate_token_logprobs(judgment, match, logprobs)
            if token_logprobs:
                avg_logprob = sum(token_logprobs) / len(token_logprobs)
        
        return {
            "match": match,
            "avg_logprob": avg_logprob
        }, False
    else:
        return None, False  # Multiple conflicting matches

def test_logprob_extraction():
    """Test logprob extraction from various judgment formats"""
    # Simple judgment with overall score
    judgment1 = "Therefore, the final verdict is:\n\nAssistant B is slightly better: [[B>A]]"
    
    # Create mock logprobs data
    # Format: (token_text, logprob)
    mock_tokens1 = [
        ("Therefore", -1.5),
        (", the ", -0.2),
        ("final", -0.4),
        (" verdict", -0.6),
        (" is:\n\n", -0.3),
        ("Assistant", -0.8),
        (" B ", -0.2),
        ("is", -0.1),
        (" slightly", -0.3),
        (" better", -0.5),
        (": ", -0.1),
        ("[[", -2.0),    # Score token
        ("B>A", -3.0),   # Score token - this is what we want to extract
        ("]]", -1.8),    # Score token
    ]
    
    mock_logprobs1 = MockLogProbs(mock_tokens1)
    
    # Run extraction
    overall_pattern = re.compile(r'\[\[([AB<>=]+)\]\]')
    result1, continue_flag1 = get_score_logprobs(judgment1, overall_pattern, mock_logprobs1)
    
    print("Test 1: Simple judgment with overall score")
    print("Judgment:", repr(judgment1))
    print("Result:", result1)
    print("Expected avg_logprob:", -3.0)  # Should equal the logprob of the token "B>A"
    print("Success:", result1["avg_logprob"] == -3.0)
    print()
    
    # Advanced test with more complex tokenization where the match spans multiple tokens
    judgment2 = "My final verdict is tie: [[A=B]]"
    
    # In this case, the pattern is split across tokens
    mock_tokens2 = [
        ("My", -0.5),
        (" final", -0.3),
        (" verdict", -0.6),
        (" is", -0.1),
        (" tie", -0.4),
        (": ", -0.1),
        ("[[", -1.8),
        ("A", -2.0),    # Part of score token
        ("=", -2.2),    # Part of score token
        ("B", -2.5),    # Part of score token
        ("]]", -1.7),
    ]
    
    mock_logprobs2 = MockLogProbs(mock_tokens2)
    
    # Run extraction
    result2, continue_flag2 = get_score_logprobs(judgment2, overall_pattern, mock_logprobs2)
    
    print("Test 2: Complex tokenization where match spans multiple tokens")
    print("Judgment:", repr(judgment2))
    print("Result:", result2)
    
    # The expected avg_logprob should be the average of the logprobs for A, =, B
    expected_avg_logprob2 = (-2.0 - 2.2 - 2.5) / 3
    print("Expected avg_logprob:", expected_avg_logprob2)
    print("Success:", result2["avg_logprob"] == expected_avg_logprob2)
    print()
    
    # Test for our custom extraction for direct string scores
    judgment3 = "Correctness: ((A>>B)). My final verdict is: [[A>>B]]"
    
    mock_tokens3 = [
        ("Correctness:", -0.5),
        (" ((", -0.8),
        ("A>>B", -2.5),  # Score token for correctness
        (")). ", -0.7),
        ("My", -0.4),
        (" final", -0.3),
        (" verdict", -0.6),
        (" is:", -0.2),
        (" [[", -1.8),
        ("A>>B", -3.5),  # Score token for overall
        ("]]", -1.7),
    ]
    
    mock_logprobs3 = MockLogProbs(mock_tokens3)
    
    # Test extraction of overall score
    result3, continue_flag3 = get_score_logprobs(judgment3, overall_pattern, mock_logprobs3)
    
    # Test custom extraction of score_logprob
    # This simulates our improved code for the "overall" score
    score_marker = f"[[{result3['match']}]]"
    token_logprobs = []
    
    token_offset = 0
    score_start = judgment3.find(score_marker)
    
    if score_start >= 0:
        score_end = score_start + len(score_marker)
        
        # Go through tokens and find those within our match
        for token in mock_logprobs3.content:
            if hasattr(token, "text") and hasattr(token, "logprob"):
                token_text = token.text
                token_length = len(token_text)
                
                # Check if this token is within our match
                token_start = token_offset
                token_end = token_start + token_length
                
                if token_start >= score_start and token_start < score_end:
                    token_logprobs.append(token.logprob)
                
                token_offset += token_length
    
    custom_avg_logprob = None
    if token_logprobs:
        custom_avg_logprob = sum(token_logprobs) / len(token_logprobs)
    
    print("Test 3: Custom score_logprob extraction for direct string score")
    print("Judgment:", repr(judgment3))
    print("Original get_score_logprobs result:", result3)
    print("Custom extraction logprobs:", token_logprobs)
    print("Custom avg_logprob:", custom_avg_logprob)
    print("Expected token logprob:", -3.5)  # Should capture the logprob of "A>>B"
    print("Success:", -3.5 in token_logprobs)

if __name__ == "__main__":
    print("=" * 60)
    print(" LOGPROB EXTRACTION TEST ")
    print("=" * 60)
    
    test_logprob_extraction()