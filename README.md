# Craft Regex


A simple, educational regular expression engine built in Rust, inspired by Russ Cox's article, ["Regular Expression Matching Can Be Simple And Fast"](https://swtch.com/~rsc/regexp/regexp1.html).

This project serves as a hands-on implementation of a Thompson's construction (NFA-based) regex engine. It parses regex patterns into an intermediate postfix notation and then compiles them into a Nondeterministic Finite Automaton (NFA) for matching.

## Features

The regex engine supports a subset of modern regex syntax:

- [x] **`*`**: Zero or more repetitions.
- [x] **`+`**: One or more repetitions.
- [x] **`?`**: Zero or one repetition (optional).
- [x] **`|`**: Alternation (e.g., `a|b`).
- [x] **`()`**: Grouping.
- [x] **`\d`**: Matches any digit (`[0-9]`).
- [x] **`[...]`**: Positive character groups (e.g., `[abc]`, `[a-z]`).
- [x] **`[^...]`**: Negative character groups (e.g., `[^abc]`)
- [x] **`^`**: Start of string anchor (e.g., `^abc`)
- [x] **`$`**: End of string anchor (e.g., `abc$`)

## Usage

1.  **Build the project:**
    ```sh
    cargo build --release
    ```

2.  **Run the engine:**

    The program reads from standard input. You can pipe text to it.

    ```sh
    echo <input_text> | ./your_program.sh -E "<pattern>"
    ```
    - The program exits with status code `0` if a match is found, and `1` otherwise.

## Implementation Details

- **Parsing**: A Shunting-yard algorithm is used to convert the infix regex pattern into a postfix (Reverse Polish Notation) representation. As it's easier to generate the Nondeterministic Finite Automaton (NFA) from a postfix expression.
- **Compilation**: The postfix expression is compiled into a NFA using Thompson's construction. Each operation in the postfix expression corresponds to a specific NFA construction step.
- **Simulation**: The NFA is simulated against the input string via `is_match` mthod to find a match. The simulation finds matches anywhere in the input strig.

