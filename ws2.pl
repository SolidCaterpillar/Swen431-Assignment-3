main :-
    current_prolog_flag(argv, Argv), member(FileName, Argv),
    atom_concat(_, '.txt', FileName), atom_concat('input-', Rest, FileName), atom_concat(Number, '.txt', Rest), atom_concat('output-', Number, OutBase), atom_concat(OutBase, '.txt', OutputFile),
    (catch(read_file_to_string(FileName, Content, []), _, fail) ->  process_stack_operations(Content, Result), open(OutputFile, write, Stream), write_output(Stream, Result), close(Stream)
    ; open(OutputFile, write, Stream), close(Stream)), halt.

% Process input and tokenize
process_stack_operations(Content, Result) :- tokenize(Content, Tokens), evaluate_stack(Tokens, [], Result).

% Main tokenizer - handle quoted strings and arrays
tokenize(Content, Tokens) :- extract_tokens(Content, RawTokens), maplist(convert_token, RawTokens, Tokens).

% Extract tokens while preserving quoted strings and arrays
extract_tokens(Content, Tokens) :- string_chars(Content, Chars), extract_tokens_chars(Chars, [], Tokens).

% Simplified token extraction
extract_tokens_chars([], Acc, Tokens) :- ( Acc = [] -> Tokens = [] ; string_chars(Token, Acc), Tokens = [Token] ).

% Handle apostrophe quoting - this must come BEFORE other special delimiters
extract_tokens_chars([''''|Rest], Acc, Tokens) :-
    (Acc = [] -> AccTokens = [] ; string_chars(Token, Acc), AccTokens = [Token]), 
    extract_next_quoted_token(Rest, QuotedToken, Remaining), extract_tokens_chars(Remaining, [], RestTokens), 
    append(AccTokens, [QuotedToken|RestTokens], Tokens).

extract_tokens_chars([Delimiter|Rest], Acc, Tokens) :-
    special_delimiter(Delimiter, ExtractPred, StartChar, EndChar),
    (Acc = [] -> AccTokens = [] ; string_chars(Token, Acc), AccTokens = [Token]),
    call(ExtractPred, Rest, Content, Remaining),
    string_concat(StartChar, Content, Temp), string_concat(Temp, EndChar, SpecialToken),
    extract_tokens_chars(Remaining, [], RestTokens),
    append(AccTokens, [SpecialToken|RestTokens], Tokens).

extract_tokens_chars([C|Rest], Acc, Tokens) :-
    \+ special_delimiter(C, _, _, _), C \= '''',  % Not an apostrophe
    ( is_whitespace(C) -> ( Acc = [] -> AccTokens = [] ; string_chars(Token, Acc), AccTokens = [Token] ), extract_tokens_chars(Rest, [], RestTokens), append(AccTokens, RestTokens, Tokens)
    ; append(Acc, [C], NewAcc), extract_tokens_chars(Rest, NewAcc, Tokens)).

% Extract the next token after an apostrophe
extract_next_quoted_token(Chars, QuotedToken, Remaining) :-
    skip_whitespace(Chars, NonWhiteChars),  % Skip any leading whitespace
    (NonWhiteChars = ['"'|Rest]   % Now extract the actual token
        % It's a quoted string
    ->  extract_quoted_string_wrapper(Rest, '"', Content, AfterString), string_concat("\"", Content, Temp), string_concat(Temp, "\"", TokenContent), string_concat("'", TokenContent, QuotedToken), Remaining = AfterString ; NonWhiteChars = ['\''|Rest]
        % It's a single-quoted string
    ->  extract_quoted_string_wrapper(Rest, '\'', Content, AfterString), string_concat("'", Content, Temp), string_concat(Temp, "'", TokenContent), string_concat("'", TokenContent, QuotedToken), Remaining = AfterString ; NonWhiteChars = ['['|Rest]
        % It's an array
    ->  extract_array(Rest, Content, AfterArray), string_concat("[", Content, Temp), string_concat(Temp, "]", TokenContent), string_concat("'", TokenContent, QuotedToken), Remaining = AfterArray
         % It's a regular token
    ; extract_regular_token(NonWhiteChars, Token, AfterToken), string_concat("'", Token, QuotedToken), Remaining = AfterToken).

% Convert string to appropriate token type
convert_token(Str, Token) :-
    string_length(Str, Len), Len >= 1, sub_string(Str, 0, 1, _, "'"), !,    % Check if the token starts with an apostrophe (quoted)
    sub_string(Str, 1, _, 0, UnquotedStr),  % Remove the apostrophe and parse the unquoted value
    parse_token_value(UnquotedStr, Value), Token = token(quoted, Value).    % Parse the unquoted string to get its actual value
convert_token(Str, Token) :- parse_token_value(Str, Value), determine_token_type(Str, Value, Token).

% Parse any string to its appropriate value
parse_token_value(Str, Value) :-
    (catch(number_string(Num, Str), _, fail) ->  Value = Num % Number
    ; Str = "true" -> Value = true ; Str = "false" -> Value = false   % Boolean
    ; string_length(Str, Len), Len >= 2, sub_string(Str, 0, 1, _, "\""), sub_string(Str, _, 1, 0, "\"")  % String (double quoted)
        ->  ContentLen is Len - 2, sub_string(Str, 1, ContentLen, 1, Content), Value = Content
    ; string_length(Str, Len), Len >= 2, sub_string(Str, 0, 1, _, "'"), sub_string(Str, _, 1, 0, "'") % String (single quoted)
        ->  ContentLen is Len - 2, sub_string(Str, 1, ContentLen, 1, Content), Value = Content
    ; string_length(Str, Len), Len >= 2, sub_string(Str, 0, 1, _, "["), sub_string(Str, _, 1, 0, "]") % Array
        ->  ContentLen is Len - 2, sub_string(Str, 1, ContentLen, 1, ArrayContent), parse_array_or_matrix(ArrayContent, Elements), Value = Elements
    ;  Value = Str  % Everything else stays as string
    ).

% Determine token type based on the original string and parsed value
determine_token_type(Str, Value, Token) :-
    (   number(Value) -> Token = token(number, Value) ; % Number
        Value = true -> Token = token(boolean, true) ; Value = false -> Token = token(boolean, false) % Boolean
    ;   string(Value), Str \= Value -> Token = token(string, Value)  % Was quoted 
    ;   string_length(Str, Len), Len >= 2, sub_string(Str, 0, 1, _, "{"), sub_string(Str, _, 1, 0, "}") % Permutation expression
    ->  ContentLen is Len - 2, sub_string(Str, 1, ContentLen, 1, PermContent), Token = token(permutation, PermContent)
    ;   is_list(Value) -> Token = token(array, Value) ; is_operator(Str) -> Token = token(operator, Str) ; Token = token(literal, Str)
    ).

is_whitespace(C) :- member(C, [' ', '\t', '\n', '\r']).

% Skip whitespace characters
skip_whitespace([], []).
skip_whitespace([C|Rest], Result) :- is_whitespace(C), !, skip_whitespace(Rest, Result).
skip_whitespace(Chars, Chars).

% Extract a regular token (until whitespace or special character)
extract_regular_token([], "", []).
extract_regular_token([C|Rest], Token, Remaining) :-
    (   is_whitespace(C)
    ->  Token = "", Remaining = [C|Rest] ; special_delimiter(C, _, _, _)
    ->  Token = "", Remaining = [C|Rest] ; C = ''''
    ->  Token = "", Remaining = [''''|Rest] ; extract_regular_token(Rest, RestToken, Remaining), string_concat(C, RestToken, Token)
    ).

% Define special delimiters and their handlers
special_delimiter('"', extract_quoted_string_wrapper('"'), '"', '"').
special_delimiter('\'', extract_quoted_string_wrapper('\''), '\'', '\'').
special_delimiter('[', extract_array, '[', ']').
special_delimiter('{', extract_permutation, '{', '}').

% Wrapper to match the expected signature
extract_quoted_string_wrapper(Quote, Rest, Content, Remaining) :-
    ( Rest = [Quote|RestAfterQuote] -> Content = "", Remaining = RestAfterQuote
    ; Rest = [C|RestChars], C \= Quote -> extract_quoted_string_wrapper(Quote, RestChars, RestContent, Remaining), string_concat(C, RestContent, Content)
    ; Rest = [] -> Content = "", Remaining = []
    ; Content = "", Remaining = Rest). % Default fallback

% Extract array content
extract_array(Chars, Content, Remaining) :- extract_array_helper(Chars, 0, [], Content, Remaining).
extract_array_helper([']'|Rest], 0, Acc, Content, Rest) :- !, string_chars(Content, Acc).
extract_array_helper([']'|Rest], Depth, Acc, Content, Remaining) :- Depth > 0, !, Depth1 is Depth - 1, append(Acc, [']'], NewAcc), extract_array_helper(Rest, Depth1, NewAcc, Content, Remaining).
extract_array_helper(['['|Rest], Depth, Acc, Content, Remaining) :- !,  Depth1 is Depth + 1, append(Acc, ['['], NewAcc), extract_array_helper(Rest, Depth1, NewAcc, Content, Remaining).
extract_array_helper([C|Rest], Depth, Acc, Content, Remaining) :- append(Acc, [C], NewAcc), extract_array_helper(Rest, Depth, NewAcc, Content, Remaining).
extract_array_helper([], _, Acc, Content, []) :- string_chars(Content, Acc).

% Extract permutation content {n | x_n-1 ... x_0}
extract_permutation(Chars, Content, Remaining) :- extract_until_char(Chars, '}', ContentChars, Remaining), string_chars(Content, ContentChars).
extract_until_char([], _, [], []). % Helper to extract until a specific character
extract_until_char([Char|Rest], Char, [], Rest) :- !.
extract_until_char([C|Rest], Char, [C|Acc], Remaining) :- extract_until_char(Rest, Char, Acc, Remaining).

% Parse array elements (handles both 1D and 2D arrays)
parse_array_or_matrix("", []) :- !.
parse_array_or_matrix(Content, Elements) :- (sub_string(Content, _, _, _, "], [") ->  parse_simple_matrix(Content, Elements) ; parse_array(Content, Elements)). % Check if content contains "], [" it's a matrix. Parse as matrix else Parse as regular array
     
% Matrix parsing, eg. "1, 2], [3, 4" split by "], ["
parse_simple_matrix(Content, Matrix) :- atom_string(ContentAtom, Content), atomic_list_concat(Parts, '], [', ContentAtom), maplist(parse_matrix_row_simple, Parts, Matrix).

% Parse each row by removing brackets and parsing as array
parse_matrix_row_simple(RowAtom, Row) :- atom_chars(RowAtom, Chars), exclude(is_bracket, Chars, CleanChars), atom_chars(CleanAtom, CleanChars), atom_string(CleanAtom, CleanStr), parse_array(CleanStr, Row).

% Helper to identify bracket characters
is_bracket('['). is_bracket(']').

% Parse 1D array elements
parse_array("", []) :- !.
parse_array(Content, Elements) :- split_string(Content, ",", " \t", StrElements), maplist(parse_element, StrElements, Elements).

parse_element(Str, Element) :- ( catch(number_string(Num, Str), _, fail) -> Element = Num ; Str = "true" -> Element = true ; Str = "false" -> Element = false ; Element = Str).

% Check if string is an operator
is_operator(Op) :- member(Op, ["+", "-", "*", "/", "**", "%", "x", "==", "!=", ">", "<", ">=", "<=", "<=>", "&", "|", "^", "<<", ">>", "!", "~", "DROP", "DUP", "SWAP", "ROT", "ROLL", "ROLLD", "IFELSE", "TRANSP", "EVAL"]).

evaluate_stack(TokenList, Stack, Result) :-
    ( TokenList = [] -> Result = Stack
    ; TokenList = [token(quoted, Value)|Tokens] -> evaluate_stack(Tokens, [Value|Stack], Result)
    ; TokenList = [token(permutation, PermStr)|Tokens] -> apply_permutation(PermStr, Stack, NewStack), evaluate_stack(Tokens, NewStack, Result)
    ; TokenList = [token(operator, Op)|Tokens] ->
        ( Op = "EVAL" -> apply_eval_operator(Stack, NewStack)
        ; member(Op, ["+", "-", "*", "/", "**", "%", "x"]) -> apply_arithmetic_operator(Op, Stack, NewStack)
        ; member(Op, ["==", "!=", ">", "<", ">=", "<=", "<=>"]) -> apply_comparison_operator(Op, Stack, NewStack)
        ; member(Op, ["&", "|", "^"]) -> apply_boolean_operator(Op, Stack, NewStack)
        ; member(Op, ["<<", ">>"]) -> apply_bitwise_operator(Op, Stack, NewStack)
        ; member(Op, ["!", "~"]) -> apply_unary_operator(Op, Stack, NewStack)
        ; Op = "IFELSE" -> apply_control_operator("IFELSE", Stack, NewStack)
        ; Op = "TRANSP" -> apply_transpose_operator(Stack, NewStack)
        ; member(Op, ["DROP", "DUP", "SWAP", "ROT", "ROLL", "ROLLD"]) -> apply_stack_operator(Op, Stack, NewStack)
        ; NewStack = Stack),  % Default case - unknown operator
        evaluate_stack(Tokens, NewStack, Result)
        ; TokenList = [token(_, Value)|Tokens] -> evaluate_stack(Tokens, [Value|Stack], Result)
        ; Result = Stack % Default case - should not happen with well-formed input
    ).

% EVAL operator - execute the top element as if it were part of input stream
apply_eval_operator([TopElement|Rest], Result) :-
    % Convert the top element to tokens and evaluate
    (   string(TopElement)
    ->  % Check if it's a permutation expression  % If It's a permutation - extract the content and apply it  %  else Regular string - tokenize and evaluate
        (sub_string(TopElement, 0, 1, _, "{"), sub_string(TopElement, _, 1, 0, "}") -> sub_string(TopElement, 1, _, 1, PermContent), apply_permutation(PermContent, Rest, Result) ; tokenize(TopElement, EvalTokens), evaluate_stack(EvalTokens, Rest, Result))
    ; atom(TopElement) -> atom_string(TopElement, TopStr), tokenize(TopStr, EvalTokens), evaluate_stack(EvalTokens, Rest, Result)
    ; Result = [TopElement|Rest] % If it's not a string or atom, just leave it on the stack
    ).
apply_eval_operator(Stack, Stack).

% Apply permutation {n | x_n-1 ... x_0 ... operations}
apply_permutation(PermStr, Stack, NewStack) :-
     % Parse the permutation string  % Tokenize the rest to get both indices and operations  % Separate index tokens from other tokens % Extract indices
    split_string(PermStr, "|", " ", [NStr, RestStr]), number_string(N, NStr), tokenize(RestStr, RestTokens), separate_indices_and_ops(RestTokens, IndexTokens, OpTokens), maplist(extract_index_from_token, IndexTokens, Indices),  
    (length(Stack, Len), N =< Len  % Apply the permutation
        ->  split_at(N, Stack, TopN, Rest),   % Store the original values for x0, x1, etc.
            reverse(TopN, ReversedTopN), % args are reversed when creating the condition
            maplist(nth0_safe(TopN), Indices, Permuted), % Apply the index permutation
            string_concat("{", PermStr, Temp), string_concat(Temp, "}", FullPermStr), % Create the full permutation string for SELF references
            evaluate_perm_ops_with_refs(OpTokens, Permuted, FullPermStr, ReversedTopN, FinalResult),  % Now evaluate any operations with the permuted elements  % Pass ReversedTopN so x0 is the first popped, x1 is second, etc.
            append(FinalResult, Rest, NewStack)
        ; NewStack = Stack  % Not enough elements
    ).

% Special evaluation that handles SELF and index references
evaluate_perm_ops_with_refs([], Stack, _, _, Stack).
evaluate_perm_ops_with_refs([token(literal, "SELF")|Tokens], Stack, PermStr, Refs, Result) :- !, evaluate_perm_ops_with_refs(Tokens, [PermStr|Stack], PermStr, Refs, Result). % Push the permutation string onto the stack as a literal that can be evaluated later
evaluate_perm_ops_with_refs([Token|Tokens], Stack, PermStr, Refs, Result) :-
    (Token = token(literal, Str), atom_string(Atom, Str), atom_concat('x', NumAtom, Atom), atom_number(NumAtom, Index)  % Check if this is an index reference that should push a value
    -> (nth0(Index, Refs, Value) ->  NewStack = [Value|Stack] ;   NewStack = Stack)  % It's an index reference - push the value from the original position  % Index out of bounds, don't push anything
    ; evaluate_stack([Token], Stack, NewStack)), evaluate_perm_ops_with_refs(Tokens, NewStack, PermStr, Refs, Result).    % Regular evaluation for other tokens

% Separate tokens that are indices (x0, x1, etc.) from other tokens
separate_indices_and_ops([], [], []).
separate_indices_and_ops([Token|Tokens], Indices, Ops) :- (is_index_token(Token) -> Indices = [Token|RestIndices], separate_indices_and_ops(Tokens, RestIndices, Ops) ; Indices = [], Ops = [Token|Tokens]). % Once we hit a non-index token, everything else is operations

% Check if a token is an index token (x0, x1, etc.)
is_index_token(token(literal, Str)) :- atom_string(Atom, Str), atom_concat('x', NumAtom, Atom), atom_number(NumAtom, _).
is_index_token(_) :- fail.

% Extract index from a token
extract_index_from_token(token(literal, Str), Index) :- atom_string(Atom, Str), atom_concat('x', NumAtom, Atom), atom_number(NumAtom, Index).

% Permute elements according to indices
% Elements are in stack order: [top, second, third, ...]
% x0 refers to top, x1 to second, etc.
nth0_safe(List, Index, Element) :- nth0(Index, List, Element), !.                               
nth0_safe(_, _, 0).

% Enhanced arithmetic operations
apply_arithmetic_operator(Op, [B, A|Rest], [Result|Rest]) :-
    (   is_matrix(A), is_matrix(B) % Check both are matrices
    ->  (Op = "*" ->  matrix_multiply(A, B, Result) ; Op = "+" -> maplist(apply_row_operation("+"), A, B, Result) ; Op = "-" -> maplist(apply_row_operation("-"), A, B, Result) ; Result = error_unsupported_matrix_op) % addition and subtraction
    ;   is_matrix(A), is_list(B), \+ is_matrix(B) -> (Op = "*" ->  matrix_vector_multiply(A, B, Result) ; Result = error_unsupported_op) % matrix * vector
    ;   is_list(A), \+ is_matrix(A), is_matrix(B) ->  (Op = "*" ->  vector_matrix_multiply(A, B, Result) ;   Result = error_unsupported_op) % vector * matrix  
    ;   is_list(A), is_list(B), same_length(A, B), \+ is_matrix(A), \+ is_matrix(B) % Check both arrays 
    ->  (Op = "*" -> maplist(calc("*"), A, B, Products), sum_list(Products, Result)  % Dot product
        ; Op = "x" -> cross_product(A, B, Result)  % Cross product  
        ; Op = "+" ->  maplist(calc("+"), A, B, Result) ; Op = "-" ->  maplist(calc("-"), A, B, Result) ;  maplist(calc(Op), A, B, Result)) % Addition: element-wise Subtraction: element-wise  Other operations
    ; Op = "+", (string(A); atom(A)), (string(B); atom(B)) ->  string_concat(A, B, Result) % String concatenation
    ; Op = "*", ((string(A); atom(A)), integer(B)) ->  repeat_string(A, B, Result) ; Op = "*", (integer(A), (string(B); atom(B))) -> repeat_string(B, A, Result) % String repetition, different position
    ; calc(Op, A, B, Result)). % Regular calculation
apply_arithmetic_operator(_, Stack, Stack).

% Enhanced calculation
calc("+", A, B, R) :- ((string(A); atom(A)), (string(B); atom(B)) -> string_concat(A, B, R) ; R is A + B).
calc("-", A, B, R) :- R is A - B.
calc("*", A, B, R) :- ((string(A); atom(A)), integer(B) ->  repeat_string(A, B, R) ; integer(A), (string(B); atom(B)) ->  repeat_string(B, A, R) ; R is A * B).
calc("/", A, B, R) :- (B =:= 0 -> R = error; R is A / B).
calc("**", A, B, R) :- R is A ** B.
calc("%", A, B, R) :- (B =:= 0 -> R = error; R is A mod B).

% Helper predicate to repeat a string N times
repeat_string(_, 0, "") :- !.
repeat_string(Str, N, Result) :- N > 0, N1 is N - 1, repeat_string(Str, N1, RestResult), string_concat(Str, RestResult, Result).
repeat_string(_, N, "") :- N < 0.

% Check if something is a matrix (2D array)
is_matrix([FirstRow|_]) :- is_list(FirstRow), !.                is_matrix(_) :- fail.

% Cross product for 3D vectors: [a1,a2,a3] x [b1,b2,b3] = [a2*b3-a3*b2, a3*b1-a1*b3, a1*b2-a2*b1]
cross_product([A1, A2, A3], [B1, B2, B3], [C1, C2, C3]) :- C1 is A2 * B3 - A3 * B2, C2 is A3 * B1 - A1 * B3, C3 is A1 * B2 - A2 * B1.

% Matrix-vector multiplication: [[a,b,c],[d,e,f]] * [x,y,z] = [ax+by+cz, dx+ey+fz]
matrix_vector_multiply(Matrix, Vector, Result) :- maplist(dot_product(Vector), Matrix, Result).

% Vector-matrix multiplication: [x,y] * [[a,b,c],[d,e,f]] = [xa+yd, xb+ye, xc+yf]
vector_matrix_multiply(Vector, Matrix, Result) :- transpose(Matrix, TransposedMatrix), maplist(dot_product(Vector), TransposedMatrix, Result).

% Matrix multiplication: A * B = C where C[i][j] = sum(A[i][k] * B[k][j]) 
matrix_multiply(A, B, C) :- transpose(B, BT), maplist(apply_row_operation(multiply, BT), A, C). % Transpose B for easier computation

dot_product(A, B, Result) :- maplist(multiply, A, B, Products), sum_list(Products, Result).

multiply(X, Y, Z) :- Z is X * Y.

% Fixed transpose predicate with proper base case handling
transpose([], []).
transpose([[]|_], []) :- !.
transpose(Matrix, [FirstCol|RestCols]) :- maplist(get_first, Matrix, FirstCol), maplist(get_rest, Matrix, RestMatrix), transpose(RestMatrix, RestCols).

get_first([H|_], H).     get_rest([_|T], T).

% Generic row operation function
apply_row_operation(multiply, BT, ARow, CRow) :- maplist(dot_product(ARow), BT, CRow).
apply_row_operation("+", RowA, RowB, RowC) :- maplist(apply_element_operation("+"), RowA, RowB, RowC).
apply_row_operation("-", RowA, RowB, RowC) :- maplist(apply_element_operation("-"), RowA, RowB, RowC).

% Generic element operation function
apply_element_operation("+", X, Y, Z) :- Z is X + Y.
apply_element_operation("-", X, Y, Z) :- Z is X - Y.

% Stack manipulation operators
apply_stack_operator("DROP", [_|Rest], Rest).
apply_stack_operator("DROP", [], []).
apply_stack_operator("DUP", [H|T], [H, H|T]).
apply_stack_operator("DUP", [], []).
apply_stack_operator("SWAP", [A, B|Rest], [B, A|Rest]).
apply_stack_operator("SWAP", Stack, Stack).
apply_stack_operator("ROT", [A, B, C|Rest], [C, A, B|Rest]).
apply_stack_operator("ROT", Stack, Stack).

% ROLL and ROLLD operations
apply_stack_operator(Op, [N|Stack], NewStack) :-
    member(Op, ["ROLL", "ROLLD"]), integer(N), N > 0, length(Stack, Len),
    (N =< Len
        -> split_at(N, Stack, Tokens, Rest), (Op = "ROLL" ->  append(Init, [Last], Tokens), append([Last|Init], Rest, NewStack) 
        ; Tokens = [Head|Tail], append(Tail, [Head], Rotated), append(Rotated, Rest, NewStack))
        ; NewStack = [N|Stack]).

% Helper predicate to split list
split_at(0, List, [], List) :- !.
split_at(N, [H|T], [H|Init], Rest) :- N > 0, N1 is N - 1, split_at(N1, T, Init, Rest).
split_at(_, [], [], []).

% Comparison operations
apply_comparison_operator(Op, [B, A|Rest], [Result|Rest]) :-
    (   Op = "==" -> (A == B -> Result = true ; Result = false)
    ;   Op = "!=" -> (A \== B -> Result = true ; Result = false)
    ;   Op = ">" -> ((number(A), number(B)) -> (A > B -> Result = true ; Result = false) ; (A @> B -> Result = true ; Result = false))
    ;   Op = "<" ->  ((number(A), number(B)) -> (A < B -> Result = true ; Result = false) ; (A @< B -> Result = true ; Result = false))
    ;   Op = ">=" ->  ((number(A), number(B)) -> (A >= B -> Result = true ; Result = false) ; (A @>= B -> Result = true ; Result = false))
    ;   Op = "<=" ->  ((number(A), number(B)) -> (A =< B -> Result = true ; Result = false) ; (A @=< B -> Result = true ; Result = false))
    ;   Op = "<=>" -> spaceship_comparison(A, B, Result)
    ;   Result = false ).
apply_comparison_operator(_, Stack, Stack).

% Spaceship operator
spaceship_comparison(A, B, Result) :- ((number(A), number(B)) ->  (A < B -> Result = -1 ; A > B -> Result = 1 ; Result = 0) ; (A @< B -> Result = -1 ;   A @> B -> Result = 1 ;   Result = 0)).

% Boolean operations
apply_boolean_operator(Op, [B, A|Rest], NewStack) :- (Op = "^" ->  xor_operation(A, B, Result), NewStack = [Result|Rest] ; boolean_operation(A, B, Op, Rest, NewStack)).
apply_boolean_operator(_, Stack, Stack).

boolean_operation(A, B, Op, Rest, NewStack) :-
    ((A = true; A = false), (B = true; B = false) 
        -> (Op = "&" -> (A = true, B = true -> Result = true ; Result = false) ; Op = "|" -> ((A = true ; B = true) -> Result = true ; Result = false) ; Result = false)
        , NewStack = [Result|Rest]
    ;   NewStack = [B, A|Rest]).

xor_operation(A, B, Result) :-
    (   (A = true; A = false), (B = true; B = false)
    ->  (A \= B -> Result = true ; Result = false)
    ;   (integer(A), integer(B))
    ->  Result is A xor B
    ;   Result = error_type_mismatch
    ).

% Bitwise operations
apply_bitwise_operator(Op, [B, A|Rest], [Result|Rest]) :-
    ((integer(A), integer(B)) ->  (   Op = "<<" -> Result is A << B ; Op = ">>" -> Result is A >> B ; Result = error_invalid_operator) ; Result = error_invalid_operands).
apply_bitwise_operator(_, Stack, Stack).

% Unary operations
apply_unary_operator(Op, [A|Rest], [Result|Rest]) :-
    (Op = "!", (A = true; A = false) -> (A = true -> Result = false ; Result = true) 
    ; Op = "~", integer(A) ->  Result is \A
    ; Result = error_invalid_operand).
apply_unary_operator(_, Stack, Stack).

% Control operations
apply_control_operator("IFELSE", [Condition, FalseVal, TrueVal|Rest], [Result|Rest]) :- ((Condition = true; Condition = false) -> (Condition = true -> Result = TrueVal ; Result = FalseVal) ; fail).
apply_control_operator(_, Stack, Stack).

% Transpose operations
apply_transpose_operator([Matrix|Rest], [Result|Rest]) :-
    ( \+ is_matrix(Matrix) -> Result = Matrix  % Non-matrix fallback
    ; Matrix = [] -> Result = []
    ; Matrix = [[]|_] -> Result = []
    ; Matrix = [[H|T]|Rows] -> maplist([Row,Head]>>(Row = [Head|_]), [[H|T]|Rows], Col), maplist([Row,Tail]>>(Row = [_|Tail]), [[H|T]|Rows], TailRows), apply_transpose_operator([TailRows], [TransposedTails]), Result = [Col|TransposedTails]
    ; Result = Matrix). % fallback
apply_transpose_operator(Stack, Stack). 

% Write output
write_output(Stream, Stack) :- reverse(Stack, ReversedStack), write_stack_elements(Stream, ReversedStack).

write_stack_elements(_, []).
write_stack_elements(Stream, [Element|Rest]) :- format_element(Element, FormattedElement), format(Stream, '~w~n', [FormattedElement]), write_stack_elements(Stream, Rest).

% Format elements
format_element(Element, Formatted) :-
    (   is_list(Element) ->  format_array(Element, Formatted)
        % Check if string is an operator - if so, output without quotes % Otherwise, add quotes
    ;   string(Element) -> (is_operator(Element) ->  Formatted = Element; atom_concat('"', Element, Temp),  atom_concat(Temp, '"', Formatted))
    ;   Formatted = Element
    ).

% Format arrays
format_array([], '[]').
format_array([H|T], Formatted) :- format_array_elements([H|T], ElementsStr), atom_concat('[', ElementsStr, Temp), atom_concat(Temp, ']', Formatted).

format_array_elements([Element], ElementStr) :- format_element(Element, ElementStr).
format_array_elements([H|T], Result) :- T \= [], format_element(H, HStr), format_array_elements(T, TStr), atom_concat(HStr, ', ', Temp), atom_concat(Temp, TStr, Result).