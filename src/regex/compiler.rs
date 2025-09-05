use bitflags::bitflags;

// The main public function of the compiler module.
// It orchestrates the three-pass compilation process.
pub fn compile(pattern: &str) -> NFA {
    let mut anchors = Anchors::empty();
    let mut pattern_slice = pattern;

    // Pass 0: Detect and strip anchors from the raw pattern string.
    if pattern_slice.starts_with('^') {
        anchors |= Anchors::START_OF_STRING;
        pattern_slice = &pattern_slice[1..];
    }
    if pattern_slice.ends_with('$') {
        anchors |= Anchors::END_OF_STRING;
        pattern_slice = &pattern_slice[..pattern_slice.len() - 1];
    }

    // Pass 1: Tokenize the pattern string into a sequence of Tokens.
    let tokens = tokenize(pattern_slice);
    // Pass 2: Insert explicit concatenation tokens where necessary.
    let tokens_with_concat = insert_concatenation(tokens);
    // Pass 3: Reorder the tokens into postfix (RPN) order using Shunting-yard.
    let postfix_tokens = shunting_yard(tokens_with_concat);

    // Compile the final token stream into an NFA.
    let mut nfa = postfix_to_nfa(&postfix_tokens);
    // Attach the detected anchors to the final NFA.
    nfa.anchors = anchors;

    nfa
}

// --- Data Structures ---

#[derive(Debug, Clone, PartialEq)]
pub enum CharacterClass {
    Digit,                                      // Represents `\d`
    Word,                                       // Represents `\w`
    Any,                                        // Represents '.'
    Set { data: Vec<char>, is_positive: bool }, // Represents `[^abc]` or `[abc]`
}

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

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Literal(char),         // A literal char like 'a'
    Class(CharacterClass), // A character class like `\\d` or `[a-z]`
    Concat,                // The internal concat operator: #
    Alternation,           // The | operator
    ZeroOrMore,            // The * operator
    OneOrMore,             // The + operator
    ZeroOrOne,             // The ? operator
    CaptureStart(usize),   // A ( mark the start of the capture group
    CaptureEnd(usize),     // A ) mark the end of the capture group
}

bitflags! {
    // Define a struct that will hold the flags.
    // `Anchors` will behave like a regular struct but with bitwise operator support.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Anchors: u8 { // u8 is enough for up to 8 flags.
        const START_OF_STRING = 0b00000001;
        const END_OF_STRING   = 0b00000010;
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

// --- NFA Simulation ---

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
        let mut current_states = self.get_epsilon_closure(vec![self.start_state]);

        if !self.anchors.contains(Anchors::END_OF_STRING) {
            if current_states.contains(&self.match_state) {
                return true;
            }
        }

        for c in input.chars() {
            let next_raw_states = self.step_state(&current_states, c);
            current_states = self.get_epsilon_closure(next_raw_states);
            current_states.extend(self.get_epsilon_closure(vec![self.start_state]));
            current_states.sort();
            current_states.dedup();

            if !self.anchors.contains(Anchors::END_OF_STRING) {
                if current_states.contains(&self.match_state) {
                    return true;
                }
            }
        }
        current_states.contains(&self.match_state)
    }
    /// Simulates the NFA against an input string to check for a match.
    /// This match the string in a stricter anchored awared way.
    fn run_match(&self, input: &str) -> bool {
        // `current_states` holds the set of all states the NFA is currently in.
        let mut current_states = self.get_epsilon_closure(vec![self.start_state]);

        if input.is_empty() {
            return current_states.contains(&self.match_state);
        }

        for c in input.chars() {
            let next_raw_states = self.step_state(&current_states, c);
            current_states = self.get_epsilon_closure(next_raw_states);

            if current_states.is_empty() {
                return false;
            }

            // If an end anchor exists, we CANNOT return early. We must consume the whole string.
            // If there is no end anchor, we can return as soon as a match is found.
            if !self.anchors.contains(Anchors::END_OF_STRING) {
                // If the set of current states includes the final match state, we have found a match.
                if current_states.contains(&self.match_state) {
                    return true;
                }
            }
        }
        current_states.contains(&self.match_state)
    }

    fn step_state(&self, current_states: &[usize], c: char) -> Vec<usize> {
        let mut next_states = Vec::new();
        for &state_idx in current_states {
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
        CharacterClass::Word => c.is_ascii_alphanumeric() || c == '_',
        CharacterClass::Any => true,
        CharacterClass::Set { data, is_positive } => {
            let found = data.binary_search(&c).is_ok();
            *is_positive == found
        }
    }
}

// --- Infix to Postfix Conversion (Shunting-yard Algorithm) ---

/// Returns the precedence of a regex operator. Higher numbers mean higher precedence.
fn precedence(token: &Token) -> u8 {
    match token {
        Token::Alternation => 1,
        Token::Concat => 2,
        Token::ZeroOrOne | Token::ZeroOrMore | Token::OneOrMore => 3,
        _ => 0,
    }
}

/// Pass 1: Scan the raw string and convert it into a vector of Tokens.
fn tokenize(pattern: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut chars = pattern.chars().peekable();
    let mut capture_group_count = 0;

    while let Some(c) = chars.next() {
        match c {
            '|' => tokens.push(Token::Alternation),
            '*' => tokens.push(Token::ZeroOrMore),
            '+' => tokens.push(Token::OneOrMore),
            '?' => tokens.push(Token::ZeroOrOne),
            '.' => tokens.push(Token::Class(CharacterClass::Any)),
            '(' => {
                capture_group_count += 1;
                tokens.push(Token::CaptureStart(capture_group_count));
            }
            ')' => tokens.push(Token::CaptureEnd(0)), // Index is adjusted later
            '[' => {
                let mut set = Vec::new();
                let positive = chars.peek() != Some(&'^');
                if !positive {
                    chars.next();
                } // Consume '^'

                while let Some(sc) = chars.next() {
                    if sc == ']' {
                        break;
                    }
                    set.push(sc);
                }
                set.sort();
                set.dedup();
                tokens.push(Token::Class(CharacterClass::Set {
                    data: set,
                    is_positive: positive,
                }));
            }
            '\\' => {
                if let Some(escaped) = chars.next() {
                    match escaped {
                        'd' => tokens.push(Token::Class(CharacterClass::Digit)),
                        'w' => tokens.push(Token::Class(CharacterClass::Word)),
                        _ => tokens.push(Token::Literal(escaped)),
                    }
                }
            }
            _ => tokens.push(Token::Literal(c)),
        }
    }
    tokens
}

/// Pass 2: Insert explicit Concat tokens where concatenation is implicit.
fn insert_concatenation(tokens: Vec<Token>) -> Vec<Token> {
    let mut result = Vec::new();
    for (i, token) in tokens.iter().enumerate() {
        result.push(token.clone());
        if i + 1 < tokens.len() {
            let next_token = &tokens[i + 1];
            let is_current_operand = matches!(
                token,
                Token::Literal(_)
                    | Token::Class(_)
                    | Token::CaptureEnd(_)
                    | Token::ZeroOrMore
                    | Token::OneOrMore
                    | Token::ZeroOrOne
            );
            let is_next_operand = matches!(
                next_token,
                Token::Literal(_) | Token::Class(_) | Token::CaptureStart(_)
            );

            if is_current_operand && is_next_operand {
                result.push(Token::Concat);
            }
        }
    }
    result
}

/// Pass 3: Reorder tokens from infix to postfix using Shunting-yard.
fn shunting_yard(tokens: Vec<Token>) -> Vec<Token> {
    let mut output = Vec::new();
    let mut operators = Vec::new();

    for token in tokens {
        match token {
            Token::Literal(_) | Token::Class(_) => output.push(token),
            Token::Concat
            | Token::Alternation
            | Token::ZeroOrMore
            | Token::OneOrMore
            | Token::ZeroOrOne => {
                while let Some(top_op) = operators.last() {
                    if matches!(top_op, Token::CaptureStart(_)) {
                        break;
                    }
                    if precedence(top_op) >= precedence(&token) {
                        output.push(operators.pop().unwrap());
                    } else {
                        break;
                    }
                }
                operators.push(token);
            }
            Token::CaptureStart(_) => operators.push(token),
            Token::CaptureEnd(_) => {
                while let Some(top_op) = operators.last() {
                    if matches!(top_op, Token::CaptureStart(_)) {
                        break;
                    }
                    output.push(operators.pop().unwrap());
                }
                if !operators.is_empty() {
                    operators.pop();
                }
            }
        }
    }

    while let Some(op) = operators.pop() {
        output.push(op);
    }

    output
}

/// Compiles a postfix expression into an NFA.
fn postfix_to_nfa(postfix_tokens: &[Token]) -> NFA {
    let mut nfa_stack: Vec<NFA> = Vec::new();

    for token in postfix_tokens {
        match token {
            Token::Literal(c) => nfa_stack.push(nfa_from_literal(*c)),
            Token::Class(class) => nfa_stack.push(nfa_from_class(class.clone())),
            Token::Concat => {
                let b = nfa_stack.pop().unwrap();
                let a = nfa_stack.pop().unwrap();
                nfa_stack.push(nfa_concat(a, b));
            }
            Token::Alternation => {
                let b = nfa_stack.pop().unwrap();
                let a = nfa_stack.pop().unwrap();
                nfa_stack.push(nfa_or(a, b));
            }
            Token::ZeroOrMore => {
                let a = nfa_stack.pop().unwrap();
                nfa_stack.push(nfa_kleene_star(a));
            }
            Token::OneOrMore => {
                let a = nfa_stack.pop().unwrap();
                nfa_stack.push(nfa_kleene_plus(a));
            }
            Token::ZeroOrOne => {
                let a = nfa_stack.pop().unwrap();
                nfa_stack.push(nfa_optional(a));
            }
            Token::CaptureStart(_) | Token::CaptureEnd(_) => {}
        }
    }

    nfa_stack.pop().unwrap_or_else(NFA::new)
}

// --- NFA Fragment Combination Functions ---

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

fn nfa_concat(mut a: NFA, mut b: NFA) -> NFA {
    let b_start_offset = a.states.len();
    a.states[a.match_state]
        .transitions
        .push((Transition::Epsilon, b.start_state + b_start_offset));
    for state in &mut b.states {
        for (_, to_idx) in &mut state.transitions {
            *to_idx += b_start_offset;
        }
    }
    a.states.append(&mut b.states);
    a.match_state = b.match_state + b_start_offset;
    a.anchors.insert(b.anchors);
    a
}

/// Combines two NFA fragments for alternation (for `a|b`).
/// This will:
/// A: [0 -> 1 -> 2 -> 3 -> 4]
/// B: [5 -> 6 -> 7]
/// => [ start -|-epsilon-> 0 -> 1 -> 2 -> 3 -> 4 -epsilon-|-> end]
///             |-epsilon-> 5 -> 6 -> 7           -epsilon-|
fn nfa_or(mut a: NFA, mut b: NFA) -> NFA {
    let mut states = vec![State::new()];
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
    states.push(State::new());
    states[0]
        .transitions
        .push((Transition::Epsilon, a.start_state + a_start_offset));
    states[0]
        .transitions
        .push((Transition::Epsilon, b.start_state + b_start_offset));
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
        anchors: a.anchors.union(b.anchors),
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
    let mut states = vec![State::new()];
    let nfa_start_offset = states.len();
    for state in &mut nfa.states {
        for (_, to_idx) in &mut state.transitions {
            *to_idx += nfa_start_offset;
        }
    }
    states.append(&mut nfa.states);
    let match_state = states.len();
    states.push(State::new());
    states[0]
        .transitions
        .push((Transition::Epsilon, nfa.start_state + nfa_start_offset));
    states[0]
        .transitions
        .push((Transition::Epsilon, match_state));
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
        anchors: nfa.anchors,
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
    let mut states = vec![State::new()];
    let nfa_start_offset = states.len();
    for state in &mut nfa.states {
        for (_, to_idx) in &mut state.transitions {
            *to_idx += nfa_start_offset;
        }
    }
    states.append(&mut nfa.states);
    let match_state = states.len();
    states.push(State::new());
    states[0]
        .transitions
        .push((Transition::Epsilon, nfa.start_state + nfa_start_offset));
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
        anchors: nfa.anchors,
    }
}

/// Creates an NFA for zero or one repetition (for `a?`).
/// This will:
/// nfa: [ 0 -> 1 -> 2 ]
///
/// => [ start -epsilon-> 0 -> 1 -> 2 -epsilon-|-> end]
///        |--------------epsilon-------------------↑
fn nfa_optional(mut nfa: NFA) -> NFA {
    let mut states = vec![State::new()];
    let nfa_start_offset = states.len();
    for state in &mut nfa.states {
        for (_, to_idx) in &mut state.transitions {
            *to_idx += nfa_start_offset;
        }
    }
    states.append(&mut nfa.states);
    let match_state = states.len();
    states.push(State::new());
    states[0]
        .transitions
        .push((Transition::Epsilon, nfa.start_state + nfa_start_offset));
    states[0]
        .transitions
        .push((Transition::Epsilon, match_state));
    states[nfa.match_state + nfa_start_offset]
        .transitions
        .push((Transition::Epsilon, match_state));
    NFA {
        states,
        start_state: 0,
        match_state,
        anchors: nfa.anchors,
    }
}
