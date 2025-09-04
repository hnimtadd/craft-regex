// Public interface of the compiler module.
// It takes a regex pattern string and returns a compiled NFA.
pub fn compile(pattern: &str) -> Nfa {
    let postfix = to_postfix(pattern);
    postfix_to_nfa(&postfix)
}

// --- Data Structures for the NFA ---

#[derive(Debug, Clone)]
pub enum CharacterClass {
    Digit,
    Word,
}

#[derive(Debug, Clone)]
pub enum Transition {
    Epsilon,
    Literal(char),
    Class(CharacterClass),
}

#[derive(Debug, Clone)]
pub struct State {
    pub transitions: Vec<(Transition, usize)>, // (Transition, to_state_index)
}

impl State {
    fn new() -> Self {
        State {
            transitions: Vec::new(),
        }
    }
}

// Represents a compiled NFA or a fragment of one.
#[derive(Debug)]
pub struct Nfa {
    pub states: Vec<State>,
    pub start_state: usize,
    pub match_state: usize,
}

impl Nfa {
    pub fn new() -> Self {
        Nfa {
            states: Vec::new(),
            start_state: 0,
            match_state: 0,
        }
    }

    pub fn is_match(&self, input: &str) -> bool {
        let mut current_states = self.get_epsilon_closure(vec![self.start_state]);

        if current_states.contains(&self.match_state) {
            return true; // Handles empty pattern matching empty string
        }

        for c in input.chars() {
            let mut next_states = Vec::new();
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

            let mut new_current_states = self.get_epsilon_closure(next_states);
            new_current_states.extend(self.get_epsilon_closure(vec![self.start_state]));
            new_current_states.sort();
            new_current_states.dedup();
            current_states = new_current_states;

            if current_states.contains(&self.match_state) {
                return true;
            }
        }

        false
    }

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

fn char_matches_class(c: char, class: &CharacterClass) -> bool {
    match class {
        CharacterClass::Digit => c.is_ascii_digit(),
        CharacterClass::Word => c.is_ascii_alphabetic(),
    }
}

// --- Infix to Postfix Conversion (Shunting-yard Algorithm) ---

fn precedence(op: char) -> u8 {
    match op {
        '|' => 1,
        '.' => 2, // Explicit concatenation operator
        '?' | '*' | '+' => 3,
        _ => 0,
    }
}

fn to_postfix(pattern: &str) -> String {
    let mut output = String::new();
    let mut operators = Vec::new();
    let mut concat_next = false;

    let mut chars = pattern.chars().peekable();

    while let Some(c) = chars.next() {
        if c.is_alphanumeric() || c == '\\' {
            if concat_next {
                while let Some(&op) = operators.last() {
                    if op != '(' && precedence(op) >= precedence('.') {
                        output.push(operators.pop().unwrap());
                    } else {
                        break;
                    }
                }
                operators.push('.');
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
                '(' => {
                    if concat_next {
                        while let Some(&op) = operators.last() {
                            if op != '(' && precedence(op) >= precedence('.') {
                                output.push(operators.pop().unwrap());
                            } else {
                                break;
                            }
                        }
                        operators.push('.');
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
                    if concat_next {
                        while let Some(&op) = operators.last() {
                            if op != '(' && precedence(op) >= precedence('.') {
                                output.push(operators.pop().unwrap());
                            } else {
                                break;
                            }
                        }
                        operators.push('.');
                    }
                    output.push(c);
                    concat_next = true;
                }
            }
        }
    }

    while let Some(op) = operators.pop() {
        output.push(op);
    }

    output
}

// --- Postfix to NFA Compilation ---

fn postfix_to_nfa(postfix: &str) -> Nfa {
    let mut nfa_stack: Vec<Nfa> = Vec::new();
    let mut chars = postfix.chars();

    while let Some(c) = chars.next() {
        match c {
            '|' => {
                let b = nfa_stack.pop().unwrap();
                let a = nfa_stack.pop().unwrap();
                nfa_stack.push(nfa_or(a, b));
            }
            '.' => {
                let b = nfa_stack.pop().unwrap();
                let a = nfa_stack.pop().unwrap();
                nfa_stack.push(nfa_concat(a, b));
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

    nfa_stack.pop().unwrap_or_else(Nfa::new)
}

// --- NFA Fragment Combination Functions ---

fn nfa_from_literal(c: char) -> Nfa {
    let mut states = vec![State::new(), State::new()];
    states[0].transitions.push((Transition::Literal(c), 1));
    Nfa {
        states,
        start_state: 0,
        match_state: 1,
    }
}

fn nfa_from_class(class: CharacterClass) -> Nfa {
    let mut states = vec![State::new(), State::new()];
    states[0].transitions.push((Transition::Class(class), 1));
    Nfa {
        states,
        start_state: 0,
        match_state: 1,
    }
}

fn nfa_concat(mut a: Nfa, mut b: Nfa) -> Nfa {
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
    a
}

fn nfa_or(mut a: Nfa, mut b: Nfa) -> Nfa {
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

    Nfa {
        states,
        start_state: 0,
        match_state,
    }
}

fn nfa_kleene_star(mut nfa: Nfa) -> Nfa {
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

    Nfa {
        states,
        start_state: 0,
        match_state,
    }
}

fn nfa_kleene_plus(mut nfa: Nfa) -> Nfa {
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

    Nfa {
        states,
        start_state: 0,
        match_state,
    }
}

fn nfa_optional(mut nfa: Nfa) -> Nfa {
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

    Nfa {
        states,
        start_state: 0,
        match_state,
    }
}

