#!/usr/bin/env python3
"""
PostToolUse Hook: Vectorization Checker

Automatically warns about for-loop anti-patterns when Python files are edited.

Installation:
1. Copy to ~/.claude/hooks/vectorization_checker.py
2. chmod +x ~/.claude/hooks/vectorization_checker.py
3. Add to settings.json (see below)

settings.json:
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "tool_name == 'Edit' || tool_name == 'Write'",
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/.claude/hooks/vectorization_checker.py \"$file_path\""
          }
        ]
      }
    ]
  }
}
"""

import sys
import re
import os
from pathlib import Path
from typing import List, Tuple

# ANSI colors
YELLOW = "\033[93m"
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"
BOLD = "\033[1m"


def check_vectorization(file_path: str) -> List[Tuple[int, str, str]]:
    """
    Check for vectorization anti-patterns.
    
    Returns:
        List of (line_number, severity, message)
    """
    issues = []
    
    if not file_path.endswith('.py'):
        return issues
    
    if not os.path.exists(file_path):
        return issues
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
    except (IOError, UnicodeDecodeError):
        return issues
    
    # Skip if disabled
    if '# vectorization-check: disable' in content:
        return issues
    
    # Patterns: (regex, severity, message)
    patterns = [
        # Nested for loops [CRITICAL]
        (
            r'for\s+\w+\s+in\s+range\s*\([^)]+\)\s*:\s*\n\s+for\s+\w+\s+in\s+range',
            'CRITICAL',
            'Nested for-loops detected. Use vectorized operations.'
        ),
        # for i in range with array access [HIGH]
        (
            r'for\s+(\w+)\s+in\s+range\s*\(\s*(?:len\s*\([^)]+\)|[a-zA-Z_]\w*)\s*\)\s*:',
            'HIGH',
            'For-loop over array length. Consider vectorization.'
        ),
        # List comprehension with function [MEDIUM]
        (
            r'\[\s*\w+\s*\(\s*\w+\s*\)\s+for\s+\w+\s+in\s+',
            'MEDIUM',
            'List comprehension on function. Consider jax.vmap.'
        ),
        # Manual accumulation [MEDIUM]
        (
            r'(\w+)\s*\+=\s*\w+\s*\[',
            'MEDIUM',
            'Manual accumulation. Consider jnp.sum/cumsum.'
        ),
        # Conditional in loop [MEDIUM]
        (
            r'if\s+\w+\s*\[\s*\w+\s*\]\s*[<>=!]',
            'MEDIUM',
            'Conditional on array element. Consider jnp.where.'
        ),
    ]
    
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        
        for pattern, severity, message in patterns:
            if '\n' in pattern:
                # Multi-line pattern
                match = re.search(pattern, content)
                if match:
                    match_line = content[:match.start()].count('\n') + 1
                    issues.append((match_line, severity, message))
            else:
                match = re.search(pattern, line)
                if match:
                    issues.append((line_num, severity, message))
    
    # Remove duplicates
    seen = set()
    unique = []
    for issue in issues:
        if issue[0] not in seen:
            seen.add(issue[0])
            unique.append(issue)
    
    return unique


def format_output(issues: List[Tuple[int, str, str]], file_path: str) -> str:
    """Format issues for terminal output."""
    if not issues:
        return ""
    
    output = [f"\n{YELLOW}⚠️  Vectorization Review: {os.path.basename(file_path)}{RESET}\n"]
    
    severity_icons = {
        'CRITICAL': f"{RED}🔴{RESET}",
        'HIGH': f"{RED}🟠{RESET}",
        'MEDIUM': f"{YELLOW}🟡{RESET}",
        'LOW': f"{GREEN}🟢{RESET}"
    }
    
    for line_num, severity, message in issues[:5]:
        icon = severity_icons.get(severity, "⚪")
        output.append(f"  {icon} Line {line_num}: {message}")
    
    if len(issues) > 5:
        output.append(f"\n  ... and {len(issues) - 5} more issues")
    
    output.append(f"\n{BOLD}Tip:{RESET} Use vectorization-reviewer agent for analysis.")
    
    return "\n".join(output)


def main():
    # Get file path from argument or stdin (Claude Code format)
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        try:
            import json
            data = json.load(sys.stdin)
            file_path = data.get('tool_input', {}).get('file_path', '')
            if not file_path:
                sys.exit(0)
        except:
            sys.exit(0)
    
    issues = check_vectorization(file_path)
    
    if issues:
        output = format_output(issues, file_path)
        print(output, file=sys.stderr)
    
    sys.exit(0)  # Don't block


if __name__ == '__main__':
    main()
