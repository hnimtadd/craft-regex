use log;
use std::{
    collections::{HashSet, VecDeque},
    fmt::{self},
};

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
    log::debug!("tokens: {:?}", tokens);
    // Pass 2: Insert explicit concatenation tokens where necessary.
    let tokens_with_concat = insert_concatenation(tokens);
    log::debug!("tokens_with_concat: {:?}", tokens_with_concat);
    // Pass 3: Reorder the tokens into postfix (RPN) order using Shunting-yard.
    let postfix_tokens = shunting_yard(tokens_with_concat);
    log::debug!("postfix: {:?}", postfix_tokens);

    // Compile the final token stream into an NFA.
    let mut nfa = postfix_to_nfa(&postfix_tokens);
    // Attach the detected anchors to the final NFA.
    nfa.anchors = anchors;

    log::debug!("nfa: {:#?}", nfa);
    nfa
}

// --- Data Structures ---

#[derive(Debug, Clone, PartialEq)]
pub enum CharacterClass {
    Digit,                                      // Represents `\d`
    Word,                                       // Represents `\w`
    Any,                                        // Represents `.`
    Set { data: Vec<char>, is_positive: bool }, // Represents `[^abc]` or `[abc]`
    BackReference(usize),                       // Represents `\1`, `\2`
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
    // A transition for the capture group.
    CaptureStart(usize),
    CaptureEnd(usize),
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
    CaptureGroup(usize),   // A () mark the end of the capture group
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

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct MatchThread {
    state_idx: usize,
    at_idx: usize, // the index in string that current thread is working on.
    capturing: Vec<CaptureGroup>, // list of capturing group
    captured: Vec<CaptureGroup>, // list of captured group
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct CaptureGroup {
    pub idx: usize,
    pub from: usize,
    pub to: usize,
}

#[derive(Debug)]
pub struct MatchResult {
    pub is_match: bool,
    pub capture_groups: Vec<CaptureGroup>,
}

/// Represents a compiled NFA. It can also be a "fragment" of a larger NFA
/// during the compilation process. Each fragment has a single start state and a single match state.
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

impl fmt::Debug for NFA {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(
            f,
            "start_state: {}, match_state: {}",
            self.start_state, self.match_state
        )
        .unwrap();
        writeln!(f, "states:").unwrap();
        for (idx, state) in self.states.iter().enumerate() {
            writeln!(f, "\t{:3}: {:?}", idx, state).unwrap();
        }
        writeln!(f, "anchors: {:?}", self.anchors).unwrap();
        Ok(())
    }
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
    pub fn simulate(&self, input: &str) -> MatchResult {
        if self.anchors.contains(Anchors::START_OF_STRING) {
            self.run_match(input)
        } else {
            self.run_search(input)
        }
    }

    /// Simulates the NFA against an input string to check for a match.
    /// This search a match if the pattern appears anywhere in the string (like `grep`).
    fn run_search(&self, input: &str) -> MatchResult {
        let mut threads = self.get_epsilon_closure(vec![MatchThread {
            state_idx: self.start_state,
            at_idx: 0,
            capturing: Vec::new(),
            captured: Vec::new(),
        }]);

        for idx in 0..input.chars().count() + 1 {
            if !self.anchors.contains(Anchors::END_OF_STRING) {
                if let Some(th) = threads.iter().find(|th| th.state_idx == self.match_state) {
                    return MatchResult {
                        is_match: true,
                        capture_groups: th.captured.clone(),
                    };
                }
            } else {
                if let Some(th) = threads.iter().find(|th| {
                    th.state_idx == self.match_state && th.at_idx == input.chars().count()
                }) {
                    return MatchResult {
                        is_match: true,
                        capture_groups: th.captured.clone(),
                    };
                }
            }

            let next_raw_threads = self.consume(threads.clone(), input);
            threads.extend(self.get_epsilon_closure(next_raw_threads));
            threads.extend(self.get_epsilon_closure(vec![MatchThread {
                at_idx: idx,
                state_idx: self.start_state,
                capturing: Vec::new(),
                captured: Vec::new(),
            }]));
            threads.sort();
            threads.dedup();
        }

        if !self.anchors.contains(Anchors::END_OF_STRING) {
            if let Some(th) = threads.iter().find(|th| th.state_idx == self.match_state) {
                return MatchResult {
                    is_match: true,
                    capture_groups: th.captured.clone(),
                };
            }
        } else {
            if let Some(th) = threads
                .iter()
                .find(|th| th.state_idx == self.match_state && th.at_idx == input.chars().count())
            {
                return MatchResult {
                    is_match: true,
                    capture_groups: th.captured.clone(),
                };
            }
        };
        MatchResult {
            is_match: false,
            capture_groups: Vec::new(),
        }
    }
    /// Simulates the NFA against an input string to check for a match.
    /// This match the string in a stricter anchored awared way.
    fn run_match(&self, input: &str) -> MatchResult {
        // `threads` holds the set of all threads the NFA is currently in.
        let mut threads = self.get_epsilon_closure(vec![MatchThread {
            state_idx: self.start_state,
            at_idx: 0,
            capturing: Vec::new(),
            captured: Vec::new(),
        }]);

        loop {
            if threads.is_empty() {
                return MatchResult {
                    is_match: false,
                    capture_groups: Vec::new(),
                };
            }

            // If an end anchor exists, we CANNOT return early. We must consume the whole string.
            // If there is no end anchor, we can return as soon as a match is found.
            if !self.anchors.contains(Anchors::END_OF_STRING) {
                // If the set of current states includes the final match state, we have found a match.
                if let Some(th) = threads.iter().find(|th| th.state_idx == self.match_state) {
                    return MatchResult {
                        is_match: true,
                        capture_groups: th.captured.clone(),
                    };
                }
            } else {
                if let Some(th) = threads.iter().find(|th| {
                    th.state_idx == self.match_state && th.at_idx == input.chars().count()
                }) {
                    return MatchResult {
                        is_match: true,
                        capture_groups: th.captured.clone(),
                    };
                }
            }

            let next_raw_threads = self.consume(threads, input);
            threads = self.get_epsilon_closure(next_raw_threads);
        }
    }

    // consume consume input at the idx with given threads state.
    fn consume(&self, threads: Vec<MatchThread>, input: &str) -> Vec<MatchThread> {
        let mut next_threads = Vec::new();
        for thread in threads.iter() {
            let Some(c) = input.chars().nth(thread.at_idx) else {
                // If the index is out of bounds, there's no character to match.
                // Return an empty vector of threads.
                continue;
            };
            for (transition, next_state_idx) in &self.states[thread.state_idx].transitions {
                match transition {
                    Transition::Literal(tc) if *tc == c => {
                        let next_thread = MatchThread {
                            state_idx: *next_state_idx,
                            at_idx: thread.at_idx + 1,
                            capturing: thread.capturing.clone(),
                            captured: thread.captured.clone(),
                        };
                        next_threads.push(next_thread);
                    }
                    Transition::Class(class) => {
                        match class {
                            CharacterClass::Any => {
                                let next_thread = MatchThread {
                                    state_idx: *next_state_idx,
                                    at_idx: thread.at_idx + 1,
                                    capturing: thread.capturing.clone(),
                                    captured: thread.captured.clone(),
                                };
                                next_threads.push(next_thread);
                            }
                            CharacterClass::Set { data, is_positive } => {
                                let found = data.binary_search(&c).is_ok();
                                if *is_positive == found {
                                    let next_thread = MatchThread {
                                        state_idx: *next_state_idx,
                                        at_idx: thread.at_idx + 1,
                                        capturing: thread.capturing.clone(),
                                        captured: thread.captured.clone(),
                                    };
                                    next_threads.push(next_thread);
                                };
                            }
                            CharacterClass::Digit => {
                                if c.is_ascii_digit() {
                                    let next_thread = MatchThread {
                                        state_idx: *next_state_idx,
                                        at_idx: thread.at_idx + 1,
                                        capturing: thread.capturing.clone(),
                                        captured: thread.captured.clone(),
                                    };
                                    next_threads.push(next_thread);
                                };
                            }
                            CharacterClass::Word => {
                                if c.is_ascii_alphanumeric() || c == '_' {
                                    let next_thread = MatchThread {
                                        state_idx: *next_state_idx,
                                        at_idx: thread.at_idx + 1,
                                        capturing: thread.capturing.clone(),
                                        captured: thread.captured.clone(),
                                    };
                                    next_threads.push(next_thread);
                                }
                            }
                            CharacterClass::BackReference(ref_idx) => thread
                                .captured
                                .iter()
                                .filter(|cg| cg.idx == *ref_idx)
                                .for_each(|cg| {
                                    log::debug!("matching for br {ref_idx}");
                                    if let (Some(slice), Some(br_slice)) = (
                                        input.get(thread.at_idx..thread.at_idx + (cg.to - cg.from)),
                                        input.get(cg.from..cg.to),
                                    ) {
                                        let is_br_match = slice == br_slice;
                                        if is_br_match {
                                            let next_thread = MatchThread {
                                                state_idx: *next_state_idx,
                                                at_idx: thread.at_idx + cg.to - cg.from,
                                                capturing: thread.capturing.clone(),
                                                captured: thread.captured.clone(),
                                            };
                                            next_threads.push(next_thread);
                                        };
                                    }
                                }),
                        };
                    }
                    _ => {}
                }
            }
        }
        next_threads
    }

    /// Computes the epsilon closure for a set of threads.
    /// An epsilon closure is the set of all threads reachable from given threads
    /// using only epsilon transitions.
    fn get_epsilon_closure(&self, threads: Vec<MatchThread>) -> Vec<MatchThread> {
        // Use the `VecDeque` as a queue for threads to visit.
        let mut to_visit: VecDeque<MatchThread> = threads.into();

        // Use a `HashSet` to keep track of visited states and avoid duplication.
        let mut visited: HashSet<MatchThread> = HashSet::from_iter(to_visit.iter().cloned());

        let mut closures = Vec::new();

        while let Some(thread) = to_visit.pop_front() {
            closures.push(thread.clone());

            // Iterate over the transition of the current thread's state.
            for (transition, next_state_idx) in &self.states[thread.state_idx].transitions {
                match transition {
                    Transition::Epsilon => {
                        let next_thread = MatchThread {
                            at_idx: thread.at_idx,
                            state_idx: *next_state_idx,
                            captured: thread.captured.clone(),
                            capturing: thread.capturing.clone(),
                        };
                        if visited.insert(next_thread.clone()) {
                            // if this one is not in visited before, add it to
                            // the queue.
                            to_visit.push_back(next_thread);
                        }
                    }
                    // at capture start, we could have 2 decisions, or step
                    // into it, or not.
                    Transition::CaptureStart(capture_group_idx) => {
                        let mut next_thread = thread.clone();
                        next_thread.capturing.push(CaptureGroup {
                            idx: *capture_group_idx,
                            from: thread.at_idx,
                            to: 0,
                        });
                        next_thread.state_idx = *next_state_idx;
                        if visited.insert(next_thread.clone()) {
                            // if this one is not in visited before, add it to
                            // the queue.
                            to_visit.push_back(next_thread);
                        }
                    }
                    Transition::CaptureEnd(capture_group_idx) => {
                        let mut next_thread = MatchThread {
                            state_idx: *next_state_idx,
                            at_idx: thread.at_idx,
                            capturing: thread.capturing.clone(),
                            captured: thread.captured.clone(),
                        };

                        // try to find the right most capturing group of
                        // capture_group_idx, then push this capture to
                        // thread.captured with updated slicing index.
                        if let Some(idx) = next_thread
                            .capturing
                            .iter()
                            .rposition(|cg| cg.idx == *capture_group_idx)
                        {
                            // this is ensured not to panic as the idx is valid at
                            // this point.
                            let cg = next_thread.capturing.get_mut(idx).unwrap();

                            // string slicing end idx is exclusive, so we have
                            // to increase idx to 1, so the capture group
                            // could capture the current index.
                            cg.to = thread.at_idx;

                            // remove the updated element from capturing and move to captured.
                            let removed_cg = next_thread.capturing.swap_remove(idx);
                            log::debug!("captured: {removed_cg:?}");
                            next_thread.captured.push(removed_cg);
                        } else {
                            panic!("capture group {} is empty", capture_group_idx);
                        }

                        if visited.insert(next_thread.clone()) {
                            // if this one is not in visited before, add it to
                            // the queue.
                            to_visit.push_back(next_thread);
                        }
                    }
                    _ => {}
                }
            }
        }
        closures
    }
}

// --- Infix to Postfix Conversion (Shunting-yard Algorithm) ---

/// Returns the precedence of a regex operator. Higher numbers mean higher precedence.
fn precedence(token: &Token) -> f32 {
    match token {
        Token::CaptureStart(idx) => 0.0 + 1.0 / (*idx as f32 + 2.0),
        Token::Alternation => 1.0,
        Token::Concat => 2.0,
        Token::ZeroOrOne | Token::ZeroOrMore | Token::OneOrMore => 3.0,
        _ => 0.0,
    }
}

/// Pass 1: Scan the raw string and convert it into a vector of Tokens.
fn tokenize(pattern: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut chars = pattern.chars().peekable();
    let mut capture_group_count = 0;
    let mut groups = Vec::new();

    while let Some(c) = chars.next() {
        match c {
            '|' => tokens.push(Token::Alternation),
            '*' => tokens.push(Token::ZeroOrMore),
            '+' => tokens.push(Token::OneOrMore),
            '?' => tokens.push(Token::ZeroOrOne),
            '.' => tokens.push(Token::Class(CharacterClass::Any)),
            '(' => {
                capture_group_count += 1;
                groups.push(capture_group_count);
                tokens.push(Token::CaptureStart(capture_group_count));
            }
            ')' => {
                let capture_group = groups.pop().unwrap();
                tokens.push(Token::CaptureEnd(capture_group));
            }
            '[' => {
                let mut set = Vec::new();
                let positive = chars.peek() != Some(&'^');
                if !positive {
                    chars.next(); // Consume '^'
                }

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
                        '1'..='9' => {
                            let mut back_reference = String::from(escaped);

                            loop {
                                if let Some(&digit) = chars.peek() {
                                    if digit.is_ascii_digit() {
                                        back_reference.push(digit);
                                        chars.next();
                                        continue;
                                    }
                                }
                                break;
                            }

                            match back_reference.parse() {
                                Ok(br) => {
                                    tokens.push(Token::Class(CharacterClass::BackReference(br)))
                                }
                                Err(err) => {
                                    panic!("invalid back reference value: {back_reference}, err: {err}")
                                }
                            };
                        }
                        _ => {}
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
    let mut buffers = Vec::new();
    let mut operators = Vec::new();

    for token in tokens {
        match token {
            Token::Literal(_) | Token::Class(_) => {
                // create new buffer if not
                let buffer = if let Some(buffer) = buffers.last_mut() {
                    buffer
                } else {
                    buffers.push(Vec::new());
                    buffers.last_mut().unwrap()
                };
                buffer.push(token)
            }
            Token::Concat
            | Token::Alternation
            | Token::ZeroOrMore
            | Token::OneOrMore
            | Token::ZeroOrOne => {
                let buffer = if let Some(buffer) = buffers.last_mut() {
                    buffer
                } else {
                    buffers.push(Vec::new());
                    buffers.last_mut().unwrap()
                };
                while let Some(op) = operators.last() {
                    if precedence(op) >= precedence(&token) {
                        buffer.push(operators.pop().unwrap());
                        continue;
                    }
                    break;
                }
                operators.push(token);
            }
            Token::CaptureStart(_) => {
                let buffer = Vec::<Token>::new();
                buffers.push(buffer);

                operators.push(token);
            }
            Token::CaptureEnd(idx) => {
                let mut buffer = buffers.pop().unwrap();
                while let Some(op) = operators.last() {
                    if matches!(op, Token::CaptureStart(_)) {
                        break;
                    }
                    buffer.push(operators.pop().unwrap());
                }
                // it is gurantee to have capture start at this point.
                assert!(matches!(operators.pop().unwrap(), Token::CaptureStart(_)));
                let last_buffer = if let Some(last_buffer) = buffers.last_mut() {
                    last_buffer
                } else {
                    buffers.push(Vec::new());
                    buffers.last_mut().unwrap()
                };
                last_buffer.extend(buffer);
                last_buffer.push(Token::CaptureGroup(idx));
            }
            Token::CaptureGroup(_) => {
                panic!("this should not happend");
            }
        }
    }

    if buffers.is_empty() {
        buffers.push(Vec::new());
    }
    let last_buffer = buffers.last_mut().unwrap();

    while let Some(op) = operators.pop() {
        last_buffer.push(op);
    }
    for buffer in buffers.iter().rev() {
        output.extend_from_slice(buffer);
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
            Token::CaptureStart(_) | Token::CaptureEnd(_) => {
                panic!("This should not happend");
            }
            Token::CaptureGroup(idx) => {
                let a = nfa_stack.pop().unwrap();
                nfa_stack.push(nfa_capture_group(*idx, a))
            }
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

/// Creates an NFA for capture start.
/// This will:
/// nfa: [ 0 -> 1 -> 2 ]
///
/// => [ start -capture_start-> 0 -> 1 -> 2 -capture_end-> end]
fn nfa_capture_group(idx: usize, mut nfa: NFA) -> NFA {
    let mut states = vec![State::new()];
    let nfa_start_offset = states.len();
    for state in &mut nfa.states {
        for (_, to_idx) in &mut state.transitions {
            *to_idx += nfa_start_offset;
        }
    }
    states.append(&mut nfa.states);

    states.push(State::new());
    let match_state = states.len() - 1;

    states[0].transitions.push((
        Transition::CaptureStart(idx),
        nfa.start_state + nfa_start_offset,
    ));

    states[nfa.match_state + nfa_start_offset]
        .transitions
        .push((Transition::CaptureEnd(idx), match_state));

    NFA {
        states,
        start_state: 0,
        match_state,
        anchors: nfa.anchors,
    }
}
