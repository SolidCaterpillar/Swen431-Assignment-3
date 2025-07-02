:- include('ws.pl').

% test_cases.pl - Test file for debugging stack calculator issues

% Include your main predicates here (copy from your main file)
% Or use :- include('your_main_file.pl'). if you want to include the main file

% Test predicates for debugging

% Test 1: Check if tokens are parsed correctly
test_token_parsing :-
    Content = "\"a\" \"b\" \"c\" \"d\" 1 2 > IFELSE",
    parse_all_tokens(Content, Tokens),
    write('Input content: '), write(Content), nl,
    write('Parsed tokens: '), write(Tokens), nl,
    length(Tokens, Len),
    write('Number of tokens: '), write(Len), nl,
    % Get the last two tokens
    append(_, [SecondLast, Last], Tokens),
    write('Second last token: '), write(SecondLast), nl,
    write('Last token: '), write(Last), nl.

% Test 2: Check if > is recognized as operator
test_operator_recognition :-
    Token = ">",
    write('Testing if '), write(Token), write(' is an operator: '),
    (is_operator_token(Token) -> 
        write('YES') ; 
        write('NO')), nl.

% Test 3: Check if IFELSE is recognized as control operator  
test_control_recognition :-
    Token = "IFELSE",
    write('Testing if '), write(Token), write(' is a control operator: '),
    (is_control_operator(Token) -> 
        write('YES') ; 
        write('NO')), nl.

% Test 4: Test individual token processing
test_individual_tokens :-
    write('=== Testing individual token processing ==='), nl,
    
    % Test number token
    process_token("1", [], Stack1),
    write('After "1": '), write(Stack1), nl,
    
    % Test another number
    process_token("2", Stack1, Stack2),
    write('After "2": '), write(Stack2), nl,
    
    % Test > operator
    process_token(">", Stack2, Stack3),
    write('After ">": '), write(Stack3), nl.

% Test 5: Test the comparison operation directly
test_comparison_direct :-
    write('=== Testing comparison directly ==='), nl,
    A = 1, B = 2, Op = ">",
    write('Testing: '), write(A), write(' '), write(Op), write(' '), write(B), nl,
    comparison_operation(A, B, Op, Result),
    write('Result: '), write(Result), nl.

% Test 6: Test apply_operator directly
test_apply_operator :-
    write('=== Testing apply_operator directly ==='), nl,
    Stack = [2, 1],
    Op = ">",
    write('Stack before: '), write(Stack), nl,
    write('Operator: '), write(Op), nl,
    apply_operator(Op, Stack, NewStack),
    write('Stack after: '), write(NewStack), nl.

% Test 7: Test IFELSE directly
test_ifelse_direct :-
    write('=== Testing IFELSE directly ==='), nl,
    Stack = [false, "d", "c", "b", "a"],
    write('Stack before IFELSE: '), write(Stack), nl,
    apply_control_operator("IFELSE", Stack, NewStack),
    write('Stack after IFELSE: '), write(NewStack), nl.

% Test 8: Full process test case 1
test_case_1 :-
    write('=== Test Case 1: 0 -9 9 1 1 == IFELSE ==='), nl,
    Content = "0 -9 9 1 1 == IFELSE",
    process_stack_operations(Content, Result),
    write('Result: '), write(Result), nl,
    write('Expected: [0, -9]'), nl.

% Test 9: Full process test case 2
test_case_2 :-
    write('=== Test Case 2: "a" "b" "c" "d" 1 2 > IFELSE ==='), nl,
    Content = "\"a\" \"b\" \"c\" \"d\" 1 2 > IFELSE",
    process_stack_operations(Content, Result),
    write('Result: '), write(Result), nl,
    write('Expected: ["a", "b", "d"]'), nl.

% Test 10: Step by step evaluation
test_step_by_step :-
    write('=== Step by step evaluation ==='), nl,
    Tokens = ["\"a\"", "\"b\"", "\"c\"", "\"d\"", "1", "2", ">", "IFELSE"],
    write('Tokens to evaluate: '), write(Tokens), nl,
    evaluate_stack(Tokens, [], Result),
    write('Final result: '), write(Result), nl.

% Test 11: Check member predicate behavior
test_member_behavior :-
    write('=== Testing member predicate behavior ==='), nl,
    Token1 = ">",
    Token2 = "IFELSE",
    OpList = ["+", "-", "*", "/", "**", "%", "==", "!=", ">", "<", ">=", "<=", "<=>", "&", "|", "^", "<<", ">>", "!", "~"],
    ControlList = ["IFELSE"],
    
    write('Testing '), write(Token1), write(' in operator list: '),
    (member(Token1, OpList) -> write('YES') ; write('NO')), nl,
    
    write('Testing '), write(Token2), write(' in control list: '),
    (member(Token2, ControlList) -> write('YES') ; write('NO')), nl.

% Run all tests
run_all_tests :-
    write('=========================================='), nl,
    write('         RUNNING ALL DEBUG TESTS         '), nl,
    write('=========================================='), nl, nl,
    
    test_token_parsing, nl,
    test_operator_recognition, nl,
    test_control_recognition, nl,
    test_individual_tokens, nl,
    test_comparison_direct, nl,
    test_apply_operator, nl,
    test_ifelse_direct, nl,
    test_case_1, nl,
    test_case_2, nl,
    test_step_by_step, nl,
    test_member_behavior, nl,
    
    write('=========================================='), nl,
    write('            TESTS COMPLETED               '), nl,
    write('=========================================='), nl.

% Quick test for the failing case
quick_test :-
    write('=== Quick test for failing case ==='), nl,
    Content = "\"a\" \"b\" \"c\" \"d\" 1 2 > IFELSE",
    parse_all_tokens(Content, Tokens),
    write('Parsed tokens: '), write(Tokens), nl,
    
    % Check if last two tokens are recognized
    append(_, [GT, IFELSE], Tokens),
    write('GT token: '), write(GT), write(' - is operator? '),
    (is_operator_token(GT) -> write('YES') ; write('NO')), nl,
    write('IFELSE token: '), write(IFELSE), write(' - is control? '),
    (is_control_operator(IFELSE) -> write('YES') ; write('NO')), nl.

% Instructions for use:
% 1. Save this as test_cases.pl
% 2. Make sure your main predicates are available (either copy them here or include your main file)
% 3. Load in SWI-Prolog: ?- [test_cases].
% 4. Run tests: ?- run_all_tests. or individual tests like ?- test_case_2.
% 5. For quick diagnosis: ?- quick_test.