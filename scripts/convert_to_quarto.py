#!/usr/bin/env python3
"""
Convert Markdown documentation to EXECUTABLE Quarto format.

Assumes YAML headers are already added manually.
Converts in-place (no directory change).
"""

import re
from pathlib import Path


def determine_doc_type(filepath: Path) -> str:
    """Determine if doc is architecture or tutorial."""
    if "Architecture" in filepath.stem:
        return "architecture"
    elif "Quick_Reference" in filepath.stem:
        return "tutorial"
    else:
        return "general"


def convert_code_blocks(content: str, doc_type: str) -> str:
    """
    Convert ```python to ```{python} with appropriate options.
    
    Strategy for architecture docs:
    - Small, demonstrative code: eval=true, output shown
    - Long code listings: eval=true, output=false (validate but don't clutter)
    - Syntax examples: eval=false (when showing invalid code patterns)
    """
    
    def replace_code_block(match):
        lang = match.group(1)
        
        # Non-Python blocks
        if lang in ['bash', 'shell', 'sh', 'yaml', 'json', 'css']:
            return f"```{{{lang}}}\n"
        
        # Python blocks - check context for hints
        preceding_context = content[max(0, match.start()-500):match.start()]
        
        # Heuristics for eval: false
        eval_false_keywords = [
            '# Bad:', '# Wrong:', '# Anti-pattern:', 
            '# Example:', '# Pseudo-code:',
            'ANTI-PATTERN', '# This is wrong', '❌'
        ]
        
        # Check if example shows WRONG code
        should_skip = any(keyword in preceding_context for keyword in eval_false_keywords)
        
        if should_skip:
            return f"```{{python}}\n#| eval: false\n"
        
        # Default: Execute but control output for architecture docs
        if doc_type == "architecture":
            # Architecture: execute but hide verbose output
            return f"```{{python}}\n#| output: false\n"
        else:
            # Tutorials/examples: show everything
            return f"```{{python}}\n"
    
    # Replace ```language with ```{language}
    content = re.sub(
        r'```(python|bash|shell|sh|yaml|json|css)\n',
        replace_code_block,
        content
    )
    
    return content


def convert_warnings_notes(content: str) -> str:
    """Convert **WARNING** and **NOTE** to Quarto callouts."""
    
    # Convert **WARNING**: ... to callout-warning
    content = re.sub(
        r'\*\*WARNING\*\*:?\s*(.+?)(?=\n\n|\n#{1,3}\s)',
        r'::: {.callout-warning}\n## Warning\n\1\n:::',
        content,
        flags=re.DOTALL
    )
    
    # Convert **NOTE**: ... to callout-note
    content = re.sub(
        r'\*\*NOTE\*\*:?\s*(.+?)(?=\n\n|\n#{1,3}\s)',
        r'::: {.callout-note}\n## Note\n\1\n:::',
        content,
        flags=re.DOTALL
    )
    
    # Convert **IMPORTANT**: ... to callout-important
    content = re.sub(
        r'\*\*IMPORTANT\*\*:?\s*(.+?)(?=\n\n|\n#{1,3}\s)',
        r'::: {.callout-important}\n## Important\n\1\n:::',
        content,
        flags=re.DOTALL
    )
    
    # Convert **TIP**: ... to callout-tip
    content = re.sub(
        r'\*\*TIP\*\*:?\s*(.+?)(?=\n\n|\n#{1,3}\s)',
        r'::: {.callout-tip}\n## Tip\n\1\n:::',
        content,
        flags=re.DOTALL
    )
    
    # Convert **CRITICAL**: ... to callout-caution
    content = re.sub(
        r'\*\*CRITICAL\*\*:?\s*(.+?)(?=\n\n|\n#{1,3}\s)',
        r'::: {.callout-caution}\n## Critical\n\1\n:::',
        content,
        flags=re.DOTALL
    )
    
    return content


def add_section_labels(content: str) -> str:
    """Add {#sec-name} labels to headers for cross-referencing."""
    
    def create_label(match):
        title = match.group(2)
        label = title.lower().replace(' ', '-').replace('/', '-')
        label = re.sub(r'[^a-z0-9-]', '', label)
        return f"{match.group(0)} {{#sec-{label}}}"
    
    # Add labels to headers (only if not already present)
    content = re.sub(
        r'^(#{2,4})\s+(.+?)(?!\s*\{#)$',
        create_label,
        content,
        flags=re.MULTILINE
    )
    
    return content


def add_executable_setup_blocks(content: str, doc_type: str) -> str:
    """
    Add setup code blocks at the start of architecture docs.
    
    This ensures all subsequent examples can execute.
    """
    
    if doc_type != "architecture":
        return content
    
    # Determine what imports are needed based on content
    needs_imports = {
        'numpy': 'import numpy as np' in content or 'np.array' in content,
        'sympy': 'import sympy as sp' in content or 'sp.symbols' in content,
        'typing': 'from typing import' in content or 'Optional[' in content,
        'systems': 'ContinuousSymbolicSystem' in content or 'DiscreteSymbolicSystem' in content,
    }
    
    # Build setup block
    setup_imports = []
    
    if needs_imports['numpy']:
        setup_imports.append('import numpy as np')
    if needs_imports['sympy']:
        setup_imports.append('import sympy as sp')
    if needs_imports['typing']:
        setup_imports.append('from typing import Optional, Tuple, Dict, List, Union')
    if needs_imports['systems']:
        setup_imports.append('from cdesym import ContinuousSymbolicSystem')
        setup_imports.append('from cdesym.systems.examples.pendulum import SymbolicPendulum')
    
    if not setup_imports:
        return content
    
    # Create setup block
    imports_str = '\n'.join(setup_imports)
    setup_block = f"""
```{{python}}
#| label: setup-imports
#| output: false
#| echo: false

# Setup for all code examples in this document
{imports_str}
```

"""
    
    # Insert after first header
    content = re.sub(
        r'(^#\s+.+?\n)',
        r'\1' + setup_block,
        content,
        count=1,
        flags=re.MULTILINE
    )
    
    return content


def convert_file(qmd_file: Path) -> None:
    """Convert single .qmd file in-place."""
    
    print(f"Converting: {qmd_file}")
    
    # Read content
    content = qmd_file.read_text()
    
    # Determine document type
    doc_type = determine_doc_type(qmd_file)
    
    # Apply transformations (NO YAML header - already done manually)
    content = convert_code_blocks(content, doc_type)
    content = convert_warnings_notes(content)
    content = add_section_labels(content)
    content = add_executable_setup_blocks(content, doc_type)
    
    # Write back in-place
    qmd_file.write_text(content)
    print(f"  ✓ Converted in-place")


def main():
    """Main conversion routine."""
    
    docs_dir = Path("docs")
    
    print("=" * 70)
    print("ControlDESymulation Documentation Conversion")
    print("In-Place Conversion (YAML headers already added)")
    print("=" * 70)
    print(f"Directory: {docs_dir}")
    print()
    
    # Find all .qmd files
    qmd_files = list(docs_dir.rglob("*.qmd"))
    
    if not qmd_files:
        print("ERROR: No .qmd files found!")
        print("Expected files with YAML headers already added.")
        exit(1)
    
    print(f"Found {len(qmd_files)} .qmd files to convert\n")
    
    # Convert each file in-place
    for qmd_file in sorted(qmd_files):
        convert_file(qmd_file)
    
    print()
    print("=" * 70)
    print("Conversion complete!")
    print(f"Converted {len(qmd_files)} files in-place")
    print()
    print("Next steps:")
    print("  1. Review changes: git diff docs/")
    print("  2. Test render: quarto preview docs/")
    print("  3. Fix any issues in code blocks")
    print("  4. Commit changes")
    print("=" * 70)


if __name__ == "__main__":
    main()