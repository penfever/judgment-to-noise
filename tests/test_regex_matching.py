#!/usr/bin/env python3
"""
Test script to verify regex pattern matching for judge outputs.
Specifically tests patterns from judge_config_multipattern.yaml for proper matching.
"""

import re
import yaml
import sys
import os
from pprint import pprint

def load_patterns(config_file):
    """Load regex patterns from config file"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    patterns = []
    if "regex_patterns" in config and config["regex_patterns"]:
        for pattern_config in config["regex_patterns"]:
            pattern_obj = {
                "name": pattern_config["name"],
                "pattern": re.compile(pattern_config["pattern"]),
                "pattern_str": pattern_config["pattern"]
            }
            patterns.append(pattern_obj)
    return patterns

def get_score(judgment, patterns):
    """Simplified version of get_score from gen_judgment.py that finds the last match"""
    scores = {}
    continue_flag = False
    
    # Process each pattern
    for pattern_obj in patterns:
        pattern_name = pattern_obj['name']
        pattern = pattern_obj['pattern']
        
        print(f"\nTesting pattern '{pattern_name}': {pattern_obj['pattern_str']}")
        matches = pattern.findall(judgment)
        print(f"Matches found: {matches}")
        
        matches = [m for m in matches if m != ""]
        
        if len(matches) == 0:
            # No matches for this pattern, continue requesting more tokens
            print(f"No matches found for pattern '{pattern_name}'")
            continue_flag = True
        elif len(matches) >= 1:
            # Get the last match instead of checking for uniqueness
            last_match = matches[-1].strip("\n")
            print(f"Found last match: '{last_match}'")
            scores[pattern_name] = last_match
        else:
            # No valid matches for this pattern
            print(f"No valid matches found")
            scores[pattern_name] = None
    
    return scores, continue_flag

def test_problematic_string():
    """Test the problematic string with the regex pattern"""
    problematic_string = """Therefore, the final verdict is:

Assistant B is slightly better: [[B>A]]"""
    
    print("Testing problematic string:")
    print("-" * 50)
    print(repr(problematic_string))
    print("-" * 50)
    
    # Direct regex test
    direct_pattern = re.compile(r'\[\[([AB<>=]+)\]\]')
    direct_match = direct_pattern.findall(problematic_string)
    print(f"Direct regex test result: {direct_match}")
    
    # Load and test with patterns from config
    config_file = os.path.join("config", "judge_config_multipattern.yaml")
    try:
        patterns = load_patterns(config_file)
        print(f"Loaded {len(patterns)} patterns from {config_file}")
        
        # Test the string with each pattern
        scores, continue_flag = get_score(problematic_string, patterns)
        
        print("\nFinal scores:")
        pprint(scores)
        print(f"Continue flag: {continue_flag}")
        
        if "overall" in scores and scores["overall"]:
            print("\n✅ SUCCESS: The overall pattern matched correctly")
        else:
            print("\n❌ FAILURE: The overall pattern did not match")
            
            # Try various transformations to debug
            print("\nDebugging transformations:")
            transformations = [
                ("Strip whitespace", problematic_string.strip()),
                ("Replace newlines with spaces", problematic_string.replace("\n", " ")),
                ("Replace newlines with \\n", repr(problematic_string)),
                ("Force ASCII-only", problematic_string.encode('ascii', 'ignore').decode())
            ]
            
            for name, transformed_str in transformations:
                print(f"\n{name}:")
                print(transformed_str)
                direct_match = direct_pattern.findall(transformed_str)
                print(f"Match result: {direct_match}")
        
    except Exception as e:
        print(f"Error loading patterns: {e}")
        return False
    
    return "overall" in scores and scores["overall"]

def test_markdown_formatted_string():
    """Test a string with markdown formatting that should still match"""
    markdown_string = """My verdicts are as follows: **Correctness**: ((A>>B)). **Completeness**: ((A=B)). **Safety**: ((A=B)). **Conciseness**: ((A=B)). **Style**: ((A=B)). My final verdict is tie: [[A=B]]"""
    
    print("\n\nTesting markdown formatted string:")
    print("-" * 50)
    print(repr(markdown_string))
    print("-" * 50)
    
    # Load and test with patterns from config
    config_file = os.path.join("config", "judge_config_multipattern.yaml")
    try:
        patterns = load_patterns(config_file)
        
        # Test the string with each pattern
        scores, continue_flag = get_score(markdown_string, patterns)
        
        print("\nFinal scores:")
        pprint(scores)
        
        success = True
        for key in ["overall", "correctness", "completeness", "safety", "conciseness", "style"]:
            if key not in scores or not scores[key]:
                print(f"\n❌ FAILURE: The {key} pattern did not match")
                success = False
        
        if success:
            print("\n✅ SUCCESS: All patterns matched correctly")
        
    except Exception as e:
        print(f"Error loading patterns: {e}")
        return False
    
    return success

def test_complex_string_with_thinking():
    """Test a complex string with thinking sections that might confuse matching"""
    complex_string = """**Final Verdict**: Assistant A is significantly better: [[A
<think>
Okay, so I need to continue my judgment based on the previous evaluation and finalize the verdict. Let me check the existing parts.

Some thinking content here with possible matches like [[A>B]] or [[A=B]].
</think>

My verdicts are as follows:  
**Correctness**: ((A>B)). Assistant A's melody adheres to folk conventions.
**Completeness**: ((A>>B)). A provides essential explanations.
**Safety**: ((A=B)). Neither answer poses risks.
**Conciseness**: ((A=B)). Both are concise for their format.
**Style**: ((A>B)). A's tone is approachable and well-organized.

**Final verdict**: Assistant A is significantly better: [[A>>B]].  

Assistant A's comprehensive explanation makes it the stronger choice."""
    
    print("\n\nTesting complex string with thinking sections:")
    print("-" * 50)
    print("(Complex string too long to display fully)")
    print("-" * 50)
    
    # Load and test with patterns from config
    config_file = os.path.join("config", "judge_config_multipattern.yaml")
    try:
        patterns = load_patterns(config_file)
        
        # Test the string with each pattern
        scores, continue_flag = get_score(complex_string, patterns)
        
        print("\nFinal scores:")
        pprint(scores)
        
        success = True
        for key in ["overall", "correctness", "completeness", "safety", "conciseness", "style"]:
            if key not in scores or not scores[key]:
                print(f"\n❌ FAILURE: The {key} pattern did not match")
                success = False
            else:
                print(f"\n✅ SUCCESS: The {key} pattern matched: {scores[key]}")
        
        if success:
            print("\n✅ SUCCESS: All patterns matched correctly")
        
        # Test overall pattern specifically for the expected final value
        if scores.get("overall") == "A>>B":
            print("✅ SUCCESS: Overall pattern correctly matched 'A>>B' as the last match")
        else:
            print(f"❌ FAILURE: Overall pattern did not match the expected value. Got: {scores.get('overall')}")
            success = False
        
    except Exception as e:
        print(f"Error loading patterns: {e}")
        return False
    
    # Test real example from the user
    real_example = """
<think>
Okay, I need to evaluate the two assistants' answers based on the user's request to write a folk tune in ABC notation. First, I should create my own answer to compare against both. Let me think about what a typical folk tune might look like. Folk tunes often have a simple structure, maybe in 4/4 time, using common scales like major or natural minor. ABC notation requires specific fields like X, T, M, L, K, and the melody itself.

Lots of thinking content with mentions of patterns like [[A>B]] and ((A=B)) which might confuse matching algorithms.
</think>

### My Answer:
Here's my own ABC notation example for a folk-style melody:

```
X:1
T:Folk Tune in D
M:4/4
L:1/8
K:D
|: D2 E2 F#2 G2 | A2 G2 F#2 E2 | D2 C2 B,2 A2 | G2 F#2 E2 D2 |
| E2 F#2 G2 A2 | B2 A2 G2 F#2 | E2 D2 C2 D2 | E2 F#2 G2 A2 :|\n|: D2 E2 F#2 G2 | A2 G2 F#2 E2 | D2 C2 B,2 A2 | G2 F#2 E2 D2 |
| E2 F#2 G2 A2 | B2 A2 G2 F#2 | E2 D2 C2 D2 | E2 F#2 G2 A2 :|\n```

### Evaluation of Assistants' Answers:

#### **Correctness**  
- **Assistant A**: Correct syntax, accurate fields (X, T, M, L, K), and proper use of repeats (`|:` and `:|`). The melody is in C major with no errors.  
- **Assistant B**: Correct syntax and structure, but the melody spans 16 bars (two 8-bar sections), which is unconventional for a folk tune (typically 8–12 bars total). The use of `B,2` (lower-octave B) is valid but stylistically unusual in folk.  
**Verdict**: ((A>B))  

#### **Completeness**  
- **Assistant A**: Includes explanations of ABC fields and structure, making it more instructive.  
- **Assistant B**: Only provides the code without context or explanation.  
**Verdict**: ((A>>B))  

#### **Safety**  
Both answers are safe and ethical.  
**Verdict**: ((A=B))  

#### **Conciseness**  
- **Assistant A**: Slightly verbose due to explanations but remains clear.  
- **Assistant B**: Extremely concise (code-only), but lacks clarity for users unfamiliar with ABC syntax.  
**Verdict**: ((A=B))  

#### **Style**  
- **Assistant A**: Friendly tone, organized with headers and bullet points.  
- **Assistant B**: Minimalist and direct but impersonal.  
**Verdict**: ((A>B))  

### Final Verdict:  
My verdicts are as follows:  
- Correctness: ((A>B)).  
- Completeness: ((A>>B)).  
- Safety: ((A=B)).  
- Conciseness: ((A=B)).  
- Style: ((A>B)).  

**Final Verdict**: Assistant A is significantly better: [[A>>B]].  

Assistant A's comprehensive explanation, adherence to folk-tune conventions, and user-friendly formatting make it the stronger choice. While Assistant B's code is valid, its brevity sacrifices instructional value and clarity."""

    print("\n\nTesting real example from user:")
    print("-" * 50)
    print("(Real example too long to display fully)")
    print("-" * 50)
    
    # Test the real example with same patterns
    scores_real, continue_flag_real = get_score(real_example, patterns)
    
    print("\nFinal scores for real example:")
    pprint(scores_real)
    
    real_success = True
    for key in ["overall", "correctness", "completeness", "safety", "conciseness", "style"]:
        if key not in scores_real or not scores_real[key]:
            print(f"\n❌ FAILURE: The {key} pattern did not match in real example")
            real_success = False
        else:
            print(f"\n✅ SUCCESS: The {key} pattern matched in real example: {scores_real[key]}")
    
    if real_success:
        print("\n✅ SUCCESS: All patterns matched correctly in real example")
    
    # Test overall pattern specifically for the expected final value
    if scores_real.get("overall") == "A>>B":
        print("✅ SUCCESS: Overall pattern correctly matched 'A>>B' as the last match in real example")
    else:
        print(f"❌ FAILURE: Overall pattern did not match the expected value in real example. Got: {scores_real.get('overall')}")
        real_success = False
    
    return success and real_success
    print("-" * 50)
    
    # Load and test with patterns from config
    config_file = os.path.join("config", "judge_config_multipattern.yaml")
    try:
        patterns = load_patterns(config_file)
        
        # Test the string with each pattern
        scores, continue_flag = get_score(complex_string, patterns)
        
        print("\nFinal scores:")
        pprint(scores)
        
        success = True
        for key in ["overall", "correctness", "completeness", "safety", "conciseness", "style"]:
            if key not in scores or not scores[key]:
                print(f"\n❌ FAILURE: The {key} pattern did not match")
                success = False
            else:
                print(f"\n✅ SUCCESS: The {key} pattern matched: {scores[key]}")
        
        if success:
            print("\n✅ SUCCESS: All patterns matched correctly")
        
        # Test overall pattern specifically for the expected final value
        if scores.get("overall") == "A>>B":
            print("✅ SUCCESS: Overall pattern correctly matched 'A>>B' as the last match")
        else:
            print(f"❌ FAILURE: Overall pattern did not match the expected value. Got: {scores.get('overall')}")
            success = False
        
    except Exception as e:
        print(f"Error loading patterns: {e}")
        return False
    
    return success

if __name__ == "__main__":
    print("=" * 60)
    print(" REGEX PATTERN MATCHING TEST ")
    print("=" * 60)
    
    test1_result = test_problematic_string()
    test2_result = test_markdown_formatted_string()
    test3_result = test_complex_string_with_thinking()
    
    if test1_result and test2_result and test3_result:
        print("\nAll tests passed successfully! ✅")
        sys.exit(0)
    else:
        print("\nSome tests failed! ❌")
        sys.exit(1)