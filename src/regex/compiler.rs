use bitflags::bitflags;

static SEPARATOR: char = '#';

// The main public function of the compiler module.
// It orchestrates the two-step compilation process:
// 1. Convert the human-readable infix regex pattern into a postfix notation (RPN).
// 2. Compile the postfix expression into a Nondeterministic Finite Automaton (NFA).
pub fn compile(pattern: &str) -> NFA {
    let mut anchors = Anchors::empty();
    let mut pattern_slice = pattern;

    // Early detect start of string and end of string anchors.
    // This is hack, but we will refactor it later.
    if pattern.starts_with('^') {
        anchors |= Anchors::START_OF_STRING;
        pattern_slice = &pattern_slice[1..];
    }
    if pattern.ends_with('$') {
        anchors |= Anchors::END_OF_STRING;
        pattern_slice = &pattern_slice[..pattern_slice.len() - 1];
    }

    let postfix = to_postfix(pattern_slice);
    println!("postfix {postfix}");

    let mut nfa = postfix_to_nfa(&postfix);
    nfa.anchors = anchors;

    nfa
}

// --- Data Structures for the NFA ---

/// Represents a class of characters. This is more general than a single literal character.
#[derive(Debug, Clone)]
pub enum CharacterClass {
    Digit,                  // Represents `\d`
    Word,                   // Represents `\w`
    Any,                    // Represents '.'
    PositiveSet(Vec<char>), // Represents `[abc]`
    NegativeSet(Vec<char>), // Represents `[^abc]`
}

/// Represents a transition between states in the NFA.
#[derive(Debug, Clone)]
pub enum Transition {
    // An epsilon transition allows the NFA to change state without consuming a character.
    // This is crucial for connecting NFA fragments for |, *, +, and ?.
    Epsilon,
    // A transition on a specific literal character.
    Literal(char),
    // A transition on a class of characters (e.g., any digit).
    Class(CharacterClass),
}

bitflags! {
    // Define a struct that will hold the flags.
    // `Anchors` will behave like a regular struct but with bitwise operator support.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Anchors: u8 { // u8 is enough for up to 8 flags
        const START_OF_STRING = 0b00000001; // Represents ^
        const END_OF_STRING   = 0b00000010; // Represents $
    }
}
/// Represents a single state in the NFA.
#[derive(Debug, Clone)]
pub struct State {
    // A state can have multiple outgoing transitions, making it non-deterministic.
    // Each tuple is a (Transition, to_state_index) pair.
    pub transitions: Vec<(Transition, usize)>,
}

impl State {
    fn new() -> Self {
        State {
            transitions: Vec::new(),
        }
    }
}

/// Represents a compiled NFA. It can also be a "fragment" of a larger NFA
/// during the compilation process. Each fragment has a single start state and a single match state.
#[derive(Debug)]
pub struct NFA {
    // All the states in the NFA.
    pub states: Vec<State>,
    // The index of the starting state in the `states` vector.
    pub start_state: usize,
    // The index of the final (or matching) state in the `states` vector.
    pub match_state: usize,
    // The anchor information of the NFA, if the state is anchored,
    // the matching process will stricter, as it only lookup from specific
    // part of the matching string.
    pub anchors: Anchors,
}

impl NFA {
    pub fn new() -> Self {
        NFA {
            states: Vec::new(),
            start_state: 0,
            match_state: 0,
            anchors: Anchors::empty(),
        }
    }
    /// Simulates the NFA against an input string to check for a match.
    pub fn is_match(&self, input: &str) -> bool {
        if self.anchors.contains(Anchors::START_OF_STRING) {
            self.run_match(input)
        } else {
            self.run_search(input)
        }
    }

    /// Simulates the NFA against an input string to check for a match.
    /// This search a match if the pattern appears anywhere in the string (like `grep`).
    fn run_search(&self, input: &str) -> bool {
        // `current_states` holds the set of all states the NFA is currently in.
        let mut current_states = self.get_epsilon_closure(vec![self.start_state]);

        // Handle patterns that can match an empty string (e.g., "a*").
        if current_states.contains(&self.match_state) {
            return true;
        }

        // Process each character of the input string.
        for c in input.chars() {
            let next_raw_states = self.step(current_states, c);

            // After processing a character, we calculate the new set of states.
            current_states = self.get_epsilon_closure(next_raw_states);

            // The key logic for substring matching.
            // After processing a character, we calculate the new set of states.
            // We also always include the NFA's main start state to allow for new matches
            // to begin at any point in the input string.
            current_states.extend(self.get_epsilon_closure(vec![self.start_state]));
            current_states.sort();
            current_states.dedup();

            // If an end anchor exists, we CANNOT return early. We must consume the whole string.
            // If there is no end anchor, we can return as soon as a match is found.
            if self.anchors.contains(Anchors::END_OF_STRING) {
                continue;
            }

            // If the set of current states includes the final match state, we have found a match.
            if current_states.contains(&self.match_state) {
                return true;
            }
        }
        // If the set of current states includes the final match state, we have found a match.
        if current_states.contains(&self.match_state) {
            return true;
        }

        // If we reach the end of the input string without finding a match.
        false
    }

    /// Simulates the NFA against an input string to check for a match.
    /// This match the string in a stricter anchored awared way.
    fn run_match(&self, input: &str) -> bool {
        // `current_states` holds the set of all states the NFA is currently in.
        let mut current_states = self.get_epsilon_closure(vec![self.start_state]);

        // Handle patterns that can match an empty string (e.g., "a*").
        if current_states.contains(&self.match_state) {
            return true;
        }

        // Process each character of the input string.
        for c in input.chars() {
            let next_raw_states = self.step(current_states, c);

            // The key logic for matching.
            // After processing a character, we calculate the new set of states.
            current_states = self.get_epsilon_closure(next_raw_states);
            if current_states.is_empty() {
                return false;
            }

            // If an end anchor exists, we CANNOT return early. We must consume the whole string.
            // If there is no end anchor, we can return as soon as a match is found.
            if self.anchors.contains(Anchors::END_OF_STRING) {
                continue;
            }

            // If the set of current states includes the final match state, we have found a match.
            if current_states.contains(&self.match_state) {
                return true;
            }
        }
        // If the set of current states includes the final match state, we have found a match.
        if current_states.contains(&self.match_state) {
            return true;
        }

        // If we reach the end of the input string without finding a match.
        false
    }

    fn step(&self, current_states: Vec<usize>, c: char) -> Vec<usize> {
        let mut next_states = Vec::new();
        // For each current state, find all possible next states based on the character `c`.
        for &state_idx in &current_states {
            for (transition, next_state_idx) in &self.states[state_idx].transitions {
                match transition {
                    Transition::Literal(tc) if *tc == c => {
                        next_states.push(*next_state_idx);
                    }
                    Transition::Class(class) => {
                        if char_matches_class(c, class) {
                            next_states.push(*next_state_idx);
                        }
                    }
                    _ => {}
                }
            }
        }
        next_states
    }

    /// Computes the epsilon closure for a set of states.
    /// An epsilon closure is the set of all states reachable from a given state
    /// using only epsilon transitions.
    fn get_epsilon_closure(&self, states: Vec<usize>) -> Vec<usize> {
        let mut closure = states;
        let mut i = 0;
        while i < closure.len() {
            let state_idx = closure[i];
            for (transition, next_state_idx) in &self.states[state_idx].transitions {
                if let Transition::Epsilon = transition {
                    if !closure.contains(next_state_idx) {
                        closure.push(*next_state_idx);
                    }
                }
            }
            i += 1;
        }
        closure
    }
}

/// Helper function for the simulator to check if a character matches a CharacterClass.
fn char_matches_class(c: char, class: &CharacterClass) -> bool {
    match class {
        CharacterClass::Digit => c.is_ascii_digit(),
        // Underscore _ is included as it is considered part of a word in
        // programming identifiers (e.g., variable and function names).
        CharacterClass::Word => c.is_ascii_alphanumeric() || c == '_',
        CharacterClass::Any => true,
        // Assumes the Vec is sorted. `binary_search` is efficient.
        CharacterClass::PositiveSet(set) => set.binary_search(&c).is_ok(),
        CharacterClass::NegativeSet(set) => set.binary_search(&c).is_err(),
    }
}

// --- Infix to Postfix Conversion (Shunting-yard Algorithm) ---

/// Returns the precedence of a regex operator. Higher numbers mean higher precedence.
fn precedence(op: char) -> u8 {
    match op {
        '|' => 1,             // Alternation
        '#' => 2,             // Concatenation (explicitly inserted)
        '?' | '*' | '+' => 3, // Quantifiers
        _ => 0,               // Not an operator
    }
}

/// Converts an infix regex pattern to a postfix (Reverse Polish Notation) expression.
/// This uses a modified Shunting-yard algorithm.
/// This step makes it much easier to compile to an NFA, as operators appear after their operands.
/// It also explicitly inserts a `.` character for concatenation (e.g., "ab" becomes "a.b").
fn to_postfix(pattern: &str) -> String {
    let mut output = String::new();
    let mut operators = Vec::new(); // Operator stack
    let mut concat_next = false;

    let mut chars = pattern.chars().peekable();

    while let Some(c) = chars.next() {
        if c.is_alphanumeric() || c == '\\' {
            if concat_next {
                // If the previous token was a literal or a group, we need to insert a concat operator.
                while let Some(&op) = operators.last() {
                    if op != '(' && precedence(op) >= precedence(SEPARATOR) {
                        output.push(operators.pop().unwrap());
                    } else {
                        break;
                    }
                }
                operators.push(SEPARATOR);
            }
            if c == '\\' {
                if let Some(escaped) = chars.next() {
                    output.push('\\');
                    output.push(escaped);
                }
            } else {
                output.push(c);
            }
            concat_next = true;
        } else {
            match c {
                '.' => {
                    if concat_next {
                        while let Some(&op) = operators.last() {
                            if op != '(' && precedence(op) >= precedence(SEPARATOR) {
                                output.push(operators.pop().unwrap());
                            } else {
                                break;
                            }
                        }
                        operators.push(SEPARATOR);
                    }
                    output.push(c);
                    concat_next = true;
                }
                '[' => {
                    if concat_next {
                        while let Some(&op) = operators.last() {
                            if op != '(' && precedence(op) >= precedence(SEPARATOR) {
                                output.push(operators.pop().unwrap());
                            } else {
                                break;
                            }
                        }
                        operators.push(SEPARATOR);
                    }
                    // Pass the entire character group through untouched.
                    output.push('[');
                    while let Some(next_c) = chars.next() {
                        output.push(next_c);
                        if next_c == ']' {
                            break;
                        }
                    }
                    concat_next = true;
                }
                '(' => {
                    if concat_next {
                        while let Some(&op) = operators.last() {
                            if op != '(' && precedence(op) >= precedence(SEPARATOR) {
                                output.push(operators.pop().unwrap());
                            } else {
                                break;
                            }
                        }
                        operators.push(SEPARATOR);
                    }
                    operators.push(c);
                    concat_next = false;
                }
                ')' => {
                    while let Some(op) = operators.pop() {
                        if op == '(' {
                            break;
                        }
                        output.push(op);
                    }
                    concat_next = true;
                }
                '|' => {
                    concat_next = false;
                    while let Some(&op) = operators.last() {
                        if op != '(' && precedence(op) >= precedence(c) {
                            output.push(operators.pop().unwrap());
                        } else {
                            break;
                        }
                    }
                    operators.push(c);
                }
                '?' | '*' | '+' => {
                    concat_next = true;
                    while let Some(&op) = operators.last() {
                        if op != '(' && precedence(op) >= precedence(c) {
                            output.push(operators.pop().unwrap());
                        } else {
                            break;
                        }
                    }
                    operators.push(c);
                }
                _ => {
                    // Any other character is treated as a literal.
                    if concat_next {
                        while let Some(&op) = operators.last() {
                            if op != '(' && precedence(op) >= precedence(SEPARATOR) {
                                output.push(operators.pop().unwrap());
                            } else {
                                break;
                            }
                        }
                        operators.push(SEPARATOR);
                    }
                    output.push(c);
                    concat_next = true;
                }
            }
        }
    }

    // Pop any remaining operators from the stack to the output.
    while let Some(op) = operators.pop() {
        output.push(op);
    }

    output
}

// --- Postfix to NFA Compilation ---

/// Parses the content of a character group (e.g., "a-z0-9").
/// It handles single characters and character ranges.
fn parse_char_group(content: &str) -> Vec<char> {
    let mut set = Vec::new();
    let mut chars = content.chars().peekable();
    while let Some(c) = chars.next() {
        // Check for a range like `a-z`.
        if let Some(&'-') = chars.peek() {
            chars.next(); // consume '-'
            if let Some(end_char) = chars.next() {
                for i in c..=end_char {
                    set.push(i);
                }
            } else {
                // Handle a trailing '-' as a literal.
                set.push(c);
                set.push('-');
            }
        } else {
            set.push(c);
        }
    }
    set.sort();
    set.dedup();
    set
}

/// Compiles a postfix expression into an NFA.
/// It processes the expression token by token, building up NFA fragments on a stack.
fn postfix_to_nfa(postfix: &str) -> NFA {
    let mut nfa_stack: Vec<NFA> = Vec::new();
    let mut chars = postfix.chars();

    while let Some(c) = chars.next() {
        match c {
            // This is our internal operator for separating segment inside
            // postfix expression.
            '#' => {
                let b = nfa_stack.pop().unwrap();
                let a = nfa_stack.pop().unwrap();
                nfa_stack.push(nfa_concat(a, b));
            }
            '[' => {
                let mut group_content = String::new();
                while let Some(next_c) = chars.next() {
                    if next_c == ']' {
                        break;
                    }
                    group_content.push(next_c);
                }

                if group_content.starts_with('^') {
                    let set = parse_char_group(&group_content[1..]);
                    nfa_stack.push(nfa_from_class(CharacterClass::NegativeSet(set)));
                } else {
                    let set = parse_char_group(&group_content);
                    nfa_stack.push(nfa_from_class(CharacterClass::PositiveSet(set)));
                }
            }
            '|' => {
                let b = nfa_stack.pop().unwrap();
                let a = nfa_stack.pop().unwrap();
                nfa_stack.push(nfa_or(a, b));
            }
            '*' => {
                let a = nfa_stack.pop().unwrap();
                nfa_stack.push(nfa_kleene_star(a));
            }
            '+' => {
                let a = nfa_stack.pop().unwrap();
                nfa_stack.push(nfa_kleene_plus(a));
            }
            '?' => {
                let a = nfa_stack.pop().unwrap();
                nfa_stack.push(nfa_optional(a));
            }
            '.' => nfa_stack.push(nfa_from_class(CharacterClass::Any)),
            '\\' => {
                if let Some(escaped_char) = chars.next() {
                    match escaped_char {
                        'd' => nfa_stack.push(nfa_from_class(CharacterClass::Digit)),
                        'w' => nfa_stack.push(nfa_from_class(CharacterClass::Word)),
                        _ => nfa_stack.push(nfa_from_literal(escaped_char)),
                    }
                }
            }
            _ => {
                nfa_stack.push(nfa_from_literal(c));
            }
        }
    }

    // The final NFA is the last one on the stack.
    nfa_stack.pop().unwrap_or_else(NFA::new)
}

// --- NFA Fragment Combination Functions ---
// These functions implement Thompson's construction.
// Each one takes one or two NFA fragments and combines them into a new, larger fragment.

/// Creates a simple NFA fragment for a single literal character.
fn nfa_from_literal(c: char) -> NFA {
    let mut states = vec![State::new(), State::new()];
    states[0].transitions.push((Transition::Literal(c), 1));
    NFA {
        states,
        start_state: 0,
        match_state: 1,
        anchors: Anchors::empty(),
    }
}

/// Creates a simple NFA fragment for a character class.
fn nfa_from_class(class: CharacterClass) -> NFA {
    let mut states = vec![State::new(), State::new()];
    states[0].transitions.push((Transition::Class(class), 1));
    NFA {
        states,
        start_state: 0,
        match_state: 1,
        anchors: Anchors::empty(),
    }
}

/// Concatenates two NFA fragments (for `ab`).
fn nfa_concat(mut a: NFA, mut b: NFA) -> NFA {
    let b_start_offset = a.states.len();
    // Add epsilon transition from the end of `a` to the start of `b`.
    a.states[a.match_state]
        .transitions
        .push((Transition::Epsilon, b.start_state + b_start_offset));

    // Before appending `b`'s states, their internal transition indices must be offset.
    for state in &mut b.states {
        for (_, to_idx) in &mut state.transitions {
            *to_idx += b_start_offset;
        }
    }

    a.states.append(&mut b.states);
    a.match_state = b.match_state + b_start_offset;
    a
}

/// Combines two NFA fragments for alternation (for `a|b`).
/// This will:
/// A: [0 -> 1 -> 2 -> 3 -> 4]
/// B: [5 -> 6 -> 7]
/// => [ start -|-epsilon-> 0 -> 1 -> 2 -> 3 -> 4 -epsilon-|-> end]
///             |-epsilon-> 5 -> 6 -> 7           -epsilon-|
fn nfa_or(mut a: NFA, mut b: NFA) -> NFA {
    let mut states = vec![State::new()]; // New start state

    let a_start_offset = states.len();
    for state in &mut a.states {
        for (_, to_idx) in &mut state.transitions {
            *to_idx += a_start_offset;
        }
    }
    states.append(&mut a.states);

    let b_start_offset = states.len();
    for state in &mut b.states {
        for (_, to_idx) in &mut state.transitions {
            *to_idx += b_start_offset;
        }
    }
    states.append(&mut b.states);

    let match_state = states.len();
    states.push(State::new()); // New match state

    // Add epsilon transitions from the new start state to the start of `a` and `b`.
    states[0]
        .transitions
        .push((Transition::Epsilon, a.start_state + a_start_offset));
    states[0]
        .transitions
        .push((Transition::Epsilon, b.start_state + b_start_offset));

    // Add epsilon transitions from the ends of `a` and `b` to the new match state.
    states[a.match_state + a_start_offset]
        .transitions
        .push((Transition::Epsilon, match_state));
    states[b.match_state + b_start_offset]
        .transitions
        .push((Transition::Epsilon, match_state));

    NFA {
        states,
        start_state: 0,
        match_state,
        anchors: Anchors::empty(),
    }
}

/// Creates an NFA for zero or more repetitions (for `a*`).
/// This will:
/// nfa: [0 -> 1 -> 2]
///
///        ↓----------------epsilon-------------------|
/// => [ start -|-epsilon-> 0 -> 1 -> 2 -epsilon-|-> end]
///        |----------------epsilon-------------------↑
fn nfa_kleene_star(mut nfa: NFA) -> NFA {
    let mut states = vec![State::new()]; // New start state
    let nfa_start_offset = states.len();

    for state in &mut nfa.states {
        for (_, to_idx) in &mut state.transitions {
            *to_idx += nfa_start_offset;
        }
    }
    states.append(&mut nfa.states);

    let match_state = states.len();
    states.push(State::new()); // New match state

    // Epsilon transition to either enter the fragment or skip it entirely.
    states[0]
        .transitions
        .push((Transition::Epsilon, nfa.start_state + nfa_start_offset));
    states[0]
        .transitions
        .push((Transition::Epsilon, match_state));

    // Epsilon transitions from the end of the fragment back to the beginning (for repetition)
    // or to the new match state.
    states[nfa.match_state + nfa_start_offset]
        .transitions
        .push((Transition::Epsilon, nfa.start_state + nfa_start_offset));
    states[nfa.match_state + nfa_start_offset]
        .transitions
        .push((Transition::Epsilon, match_state));

    NFA {
        states,
        start_state: 0,
        match_state,
        anchors: Anchors::empty(),
    }
}

/// Creates an NFA for one or more repetitions (for `a+`).
/// This will:
/// nfa: [ 0 -> 1 -> 2 ]
///
///        ↓---------------------------epsilon-------------------|
/// => [ start -(at-least-1)-epsilon-> 0 -> 1 -> 2 -epsilon-|-> end]
///        |---------------------------epsilon-------------------↑
fn nfa_kleene_plus(mut nfa: NFA) -> NFA {
    let mut states = vec![State::new()]; // New start state
    let nfa_start_offset = states.len();

    for state in &mut nfa.states {
        for (_, to_idx) in &mut state.transitions {
            *to_idx += nfa_start_offset;
        }
    }
    states.append(&mut nfa.states);

    let match_state = states.len();
    states.push(State::new()); // New match state

    // Must enter the fragment at least once.
    states[0]
        .transitions
        .push((Transition::Epsilon, nfa.start_state + nfa_start_offset));

    // Loop back for more repetitions or exit to the match state.
    states[nfa.match_state + nfa_start_offset]
        .transitions
        .push((Transition::Epsilon, nfa.start_state + nfa_start_offset));
    states[nfa.match_state + nfa_start_offset]
        .transitions
        .push((Transition::Epsilon, match_state));

    NFA {
        states,
        start_state: 0,
        match_state,
        anchors: Anchors::empty(),
    }
}

/// Creates an NFA for zero or one repetition (for `a?`).
/// This will:
/// nfa: [ 0 -> 1 -> 2 ]
///
/// => [ start -epsilon-> 0 -> 1 -> 2 -epsilon-|-> end]
///        |--------------epsilon-------------------↑
fn nfa_optional(mut nfa: NFA) -> NFA {
    let mut states = vec![State::new()]; // New start state
    let nfa_start_offset = states.len();

    for state in &mut nfa.states {
        for (_, to_idx) in &mut state.transitions {
            *to_idx += nfa_start_offset;
        }
    }
    states.append(&mut nfa.states);

    let match_state = states.len();
    states.push(State::new()); // New match state

    // Choice to enter the fragment or skip it.
    states[0]
        .transitions
        .push((Transition::Epsilon, nfa.start_state + nfa_start_offset));
    states[0]
        .transitions
        .push((Transition::Epsilon, match_state));

    // From the end of the fragment, go to the new match state.
    states[nfa.match_state + nfa_start_offset]
        .transitions
        .push((Transition::Epsilon, match_state));

    NFA {
        states,
        start_state: 0,
        match_state,
        anchors: Anchors::empty(),
    }
}
