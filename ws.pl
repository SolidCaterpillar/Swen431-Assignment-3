:- use_module(library(filesex)).
:- use_module(library(readutil)).

main :-
    current_prolog_flag(argv, Argv), 
    member(FileName, Argv), atom_concat(_, '.txt', FileName),
    file_base_name(FileName, BaseName), atom_concat('input-', Rest, BaseName), atom_concat(Number, '.txt', Rest),   % Get the base name
    atom_concat('output-', Number, OutBase), atom_concat(OutBase, '.txt', OutputFile),  % Create output filename
    (catch(read_file_to_string(FileName, Content, []), _, fail) ->      % Process the file
        process_content(Content, Result), open(OutputFile, write, Stream), write_output(Stream, Result), close(Stream)
    ; open(OutputFile, write, Stream), close(Stream)), 
    halt.

process_content(Content, Result) :- process_token(Content, Tokens), evaluate_stack(Tokens, [], Result). % Process input and tokenise

process_token(Content, Tokens) :- extract_token(Content, RawTokens), maplist(convert_token, RawTokens, Tokens). % Main tokeniser

extract_token(Content, Tokens) :- string_chars(Content, Chars), extract_chars(Chars, [], Tokens). % Extract tokens while preserving strings with quotation and arrays

extract_chars([], Acc, Tokens) :- ( Acc = [] -> Tokens = [] ; string_chars(Token, Acc), Tokens = [Token]). % Token extraction

% Handle apostrophe
extract_chars([''''|Rest], Acc, Tokens) :-
    (Acc = [] -> Current = [] ; string_chars(Token, Acc), Current = [Token]),   % Finish accumulated token
    after_apostrophe(Rest, Quoted, After),      % Extract quoted token
    extract_chars(After, [], More),   % Process remaining chars
    append(Current, [Quoted|More], Tokens).    % Combine all tokens

% Handle quotes, brackets and braces
extract_chars([Delim|Rest], Acc, Tokens) :-
    delimiter(Delim, Extract, Start, End),  % Check if delimiter
    (Acc = [] -> Current = [] ; string_chars(Token, Acc), Current = [Token]),   % Finish accumulated token
    call(Extract, Rest, Content, After),    % Extract content using predicate
    string_concat(Start, Content, Temp), string_concat(Temp, End, Special),    % Wrap content with delimiters
    extract_chars(After, [], More),
    append(Current, [Special|More], Tokens).

% Handle regular characters and whitespace
extract_chars([C|Rest], Acc, Tokens) :-
    \+ delimiter(C, _, _, _), C \= '''',  % Not a delimiter or apostrophe
     % Whitespace to finish current token
    (member(C, [' ', '\t', '\n', '\r']) -> (Acc = [] -> Current = [] ; string_chars(Token, Acc), Current = [Token]), extract_chars(Rest, [], More), append(Current, More, Tokens)
    ; append(Acc, [C], NewAcc), extract_chars(Rest, NewAcc, Tokens) % Regular character add to accumulator
    ).

% Extract the next token after an apostrophe and mark it as quoted
after_apostrophe(Chars, Quoted, After) :-
    skip_whitespace(Chars, Clean),  % Skip any whitespace
    (Clean = ['"'|Rest] ->
        % Quoted string                                                                                                  % Add apostrophe marker                          
        extract_quoted(Rest, '"', Content, Next), string_concat("\"", Content, Temp), string_concat(Temp, "\"", Token), string_concat("'", Token, Quoted), After = Next
    ; Clean = ['\''|Rest] ->
        % Single-quoted string
        extract_quoted(Rest, '\'', Content, Next), string_concat("'", Content, Temp), string_concat(Temp, "'", Token), string_concat("'", Token, Quoted), After = Next
    ; Clean = ['['|Rest] ->
        % Array
        extract_array(Rest, Content, Next), string_concat("[", Content, Temp), string_concat(Temp, "]", Token), string_concat("'", Token, Quoted), After = Next
    ; % Regular token
        extract_regular_token(Clean, Token, Next), string_concat("'", Token, Quoted), After = Next).

% Extract a regular token
extract_regular_token([], "", []).
extract_regular_token([C|Rest], Token, After) :-
    (member(C, [' ', '\t', '\n', '\r']) -> Token = "", After = [C|Rest] 
    ; delimiter(C, _, _, _) -> Token = "", After = [C|Rest] 
    ; C = '''' -> Token = "", After = [''''|Rest] 
    ; extract_regular_token(Rest, More, After), string_concat(C, More, Token)).

% Skip whitespace characters
skip_whitespace([], []).
skip_whitespace([C|Rest], Result) :- member(C, [' ', '\t', '\n', '\r']), !, skip_whitespace(Rest, Result).
skip_whitespace(Chars, Chars).

% Define special delimiters and their handlers
delimiter('"', extract_quoted('"'), '"', '"').
delimiter('\'', extract_quoted('\''), '\'', '\'').
delimiter('[', extract_array, '[', ']').
delimiter('{', extract_lambda, '{', '}').

% Wrapper to match the expected signature
extract_quoted(Quote, Rest, Content, After) :-
    (Rest = [Quote|Next] -> Content = "", After = Next
    ; Rest = [C|More], C \= Quote -> extract_quoted(Quote, More, RestContent, After), string_concat(C, RestContent, Content)
    ; Rest = [] -> Content = "", After = []
    ; Content = "", After = Rest). % Fallback

% Convert string token to token type
convert_token(Str, Token) :-    % Handles quoted tokens
    string_length(Str, Len), Len >= 1, sub_string(Str, 0, 1, _, "'"), !,    % Check if token starts with apostrophe
    sub_string(Str, 1, _, 0, Raw), parse_token(Raw, Value), Token = token(quoted, Value). % Remove the apostrophe and parse the unquoted token to get actual value

convert_token(Str, Token) :- 
    parse_token(Str, Value), token_type(Str, Value, Token). % Handle regular tokens

% Parse any string token to its appropriate value
parse_token(Str, Value) :-
    (catch(number_string(Num, Str), _, fail) -> Value = Num % Number
    ; Str = "true" -> Value = true ; Str = "false" -> Value = false   % Boolean
    ; string_length(Str, Len), Len >= 2, sub_string(Str, 0, 1, _, "\""), sub_string(Str, _, 1, 0, "\"") ->
        % String (double quoted)
        Size is Len - 2, sub_string(Str, 1, Size, 1, Content), Value = Content
    ; string_length(Str, Len), Len >= 2, sub_string(Str, 0, 1, _, "'"), sub_string(Str, _, 1, 0, "'") ->
        % String (single quoted)
        Size is Len - 2, sub_string(Str, 1, Size, 1, Content), Value = Content
    ; string_length(Str, Len), Len >= 2, sub_string(Str, 0, 1, _, "["), sub_string(Str, _, 1, 0, "]") ->
        % Array
        Size is Len - 2, sub_string(Str, 1, Size, 1, Inner), parse_array_or_matrix(Inner, Elements), Value = Elements
    ; Value = Str). % Rest stays as string
    
% Check token type and parsed value
token_type(Str, Value, Token) :-
    (number(Value) -> Token = token(number, Value)  % Number
    ; Value = true -> Token = token(boolean, true) ; Value = false -> Token = token(boolean, false) % Boolean
    ; string(Value), Str \= Value -> Token = token(string, Value)  % Was quoted
    ; string_length(Str, Len), Len >= 2, sub_string(Str, 0, 1, _, "{"), sub_string(Str, _, 1, 0, "}") ->
        Size is Len - 2, sub_string(Str, 1, Size, 1, Inner), Token = token(permutation, Inner) % Lambda expression
    ; is_list(Value) -> Token = token(array, Value) % Array
    ; is_operator(Str) -> Token = token(operator, Str) % Operator 
    ; Token = token(literal, Str)). % String

% Reference from haskell assignment's balanced delimiters recursion method and https://stackoverflow.com/questions/65151064/valid-bracket-list-in-prolog 
% Extract array content between brackets with proper nesting support using recursion similar 
extract_array(Chars, Content, After) :- extract_array_rec(Chars, 0, [], Content, After).
extract_array_rec([']'|Rest], 0, Acc, Content, Rest) :- !, string_chars(Content, Acc).
extract_array_rec([']'|Rest], Depth, Acc, Content, After) :- Depth > 0, !, Next is Depth - 1, append(Acc, [']'], New), extract_array_rec(Rest, Next, New, Content, After).
extract_array_rec(['['|Rest], Depth, Acc, Content, After) :- !, Next is Depth + 1, append(Acc, ['['], New), extract_array_rec(Rest, Next, New, Content, After).
extract_array_rec([C|Rest], Depth, Acc, Content, After) :- append(Acc, [C], New), extract_array_rec(Rest, Depth, New, Content, After).
extract_array_rec([], _, Acc, Content, []) :- string_chars(Content, Acc).

% Lambda Extraction. Format: {N | x1 x0 x2} rearranges top N stack elements according to indices
extract_lambda(Chars, Content, After) :- lambda_content(Chars, '}', List, After), string_chars(Content, List).
lambda_content([], _, [], []). % Helper to extract until a specific character
lambda_content([Char|Rest], Char, [], Rest) :- !.
lambda_content([C|Rest], End, [C|Acc], After) :- lambda_content(Rest, End, Acc, After).

% Parse array/matrix
parse_array_or_matrix("", []) :- !.
parse_array_or_matrix(Content, Elements) :-
    (sub_string(Content, _, _, _, "], [") -> atom_string(Atom, Content), atomic_list_concat(Parts, '], [', Atom), maplist(parse_matrix_row, Parts, Elements)   % Matrix
    ; split_string(Content, ",", " \t", Strings), maplist(parse_array, Strings, Elements)).   % Array

% Parse matrix row by removing brackets and parsing as comma-separated values
parse_matrix_row(RowAtom, Row) :-
    atom_chars(RowAtom, Chars), exclude([C]>>(C = '['; C = ']'), Chars, Clean), atom_chars(Atom, Clean), atom_string(Atom, Str), 
    split_string(Str, ",", " \t", Parts), maplist(parse_array, Parts, Row).

% Parse individual token in array (number, boolean, or string)
parse_array(Str, Element) :-
    (catch(number_string(Num, Str), _, fail) -> Element = Num
    ; Str = "true" -> Element = true ; Str = "false" -> Element = false
    ; Element = Str).

% Check if string is an operator
is_operator(Op) :- member(Op, ["+", "-", "*", "/", "**", "%", "x", "==", "!=", ">", "<", ">=", "<=", "<=>", "&", "|", "^", "<<", ">>", "!", "~", "DROP", "DUP", "SWAP", "ROT", "ROLL", "ROLLD", "IFELSE", "TRANSP", "EVAL"]).

evaluate_stack(Tokens, Stack, Result) :-
    (Tokens = [] -> Result = Stack
    ; Tokens = [token(quoted, Value)|Rest] -> evaluate_stack(Rest, [Value|Stack], Result)
    ; Tokens = [token(permutation, Perm)|Rest] -> lambda_operator(Perm, Stack, New), evaluate_stack(Rest, New, Result)
    ; Tokens = [token(operator, Op)|Rest] ->
        (member(Op, ["+", "-", "*", "/", "**", "%", "x"]) -> arithmetic_operator(Op, Stack, New)
        ; member(Op, ["DROP", "DUP", "SWAP", "ROT", "ROLL", "ROLLD"]) -> stack_operator(Op, Stack, New)
        ; member(Op, ["==", "!=", ">", "<", ">=", "<=", "<=>"]) -> comparison_operator(Op, Stack, New)
        ; member(Op, ["&", "|", "^"]) -> boolean_operator(Op, Stack, New)
        ; member(Op, ["<<", ">>"]) -> bitwise_operator(Op, Stack, New)
        ; member(Op, ["!", "~"]) -> unary_operator(Op, Stack, New)
        ; Op = "IFELSE" -> ifelse_operator("IFELSE", Stack, New)
        ; Op = "TRANSP" -> transpose_operator(Stack, New)
        ; Op = "EVAL" -> eval_operator(Stack, New)
        ; New = Stack), evaluate_stack(Rest, New, Result)
    ; Tokens = [token(_, Value)|Rest] -> evaluate_stack(Rest, [Value|Stack], Result)
    ; Result = Stack).

% Arithmetic operation
arithmetic_operator(Op, [B, A|Rest], [Result|Rest]) :-
    (is_list(A), A = [RowA|_], is_list(RowA), is_list(B), B = [RowB|_], is_list(RowB) ->   
        (Op = "*" -> matrix_multiply(A, B, Result)   % Both are matrices
        ; Op = "+" -> maplist([Ra,Rb,Rc]>>maplist([X,Y,Z]>>(Z is X + Y), Ra, Rb, Rc), A, B, Result)
        ; Op = "-" -> maplist([Ra,Rb,Rc]>>maplist([X,Y,Z]>>(Z is X - Y), Ra, Rb, Rc), A, B, Result)
        ; Result = error)
    ; is_list(A), A = [RowA|_], is_list(RowA), is_list(B), \+ (B = [RowB|_], is_list(RowB)) ->  
        (Op = "*" -> matrix_vector_multiply(A, B, Result)   % A is matrix, B is vector
        ; Result = error)
    ; is_list(A), \+ (A = [RowA|_], is_list(RowA)), is_list(B), B = [RowB|_], is_list(RowB) -> 
        (Op = "*" -> vector_matrix_multiply(A, B, Result)   % A is vector, B is matrix
        ; Result = error)
    ; is_list(A), is_list(B), same_length(A, B), \+ (A = [RowA|_], is_list(RowA)), \+ (B = [RowB|_], is_list(RowB)) -> 
        (Op = "*" -> dot_product(A, B, Result)  % Both vectors
        ; Op = "x" -> cross_product(A, B, Result)
        ; Op = "+" -> maplist(calc("+"), A, B, Result)
        ; Op = "-" -> maplist(calc("-"), A, B, Result)
        ; maplist(calc(Op), A, B, Result))  % Other operations
    ; Op = "+", (string(A); atom(A)), (string(B); atom(B)) -> string_concat(A, B, Result)   % String concatenation
    ; Op = "*", (string(A); atom(A)), integer(B) -> repeat_string(A, B, Result)   % String repetition (string * number)
    ; Op = "*", integer(A), (string(B); atom(B)) -> repeat_string(B, A, Result)  % String repetition (number * string)
    ; calc(Op, A, B, Result)).  % Regular calculation
arithmetic_operator(_, Stack, Stack).

% Calculation helper method for arithmetic operation
calc("+", A, B, R) :- ((string(A); atom(A)), (string(B); atom(B)) -> string_concat(A, B, R) ; R is A + B).
calc("-", A, B, R) :- R is A - B.
calc("*", A, B, R) :- ((string(A); atom(A)), integer(B) ->  repeat_string(A, B, R) ; integer(A), (string(B); atom(B)) ->  repeat_string(B, A, R) ; R is A * B).
calc("/", A, B, R) :- (B =:= 0 -> R = error ; integer(A), integer(B) -> R is A // B ; R is A / B). 
calc("**", A, B, R) :- R is A ** B.
calc("%", A, B, R) :- (B =:= 0 -> R = error; R is A mod B).

% Repeat string helper method for string multiplier
repeat_string(_, 0, "") :- !.
repeat_string(Str, N, Result) :- N > 0, Next is N - 1, repeat_string(Str, Next, Rest), string_concat(Str, Rest, Result).
repeat_string(_, N, "") :- N < 0.

% Cross product for 3D vectors: [a1,a2,a3] x [b1,b2,b3] = [a2*b3-a3*b2, a3*b1-a1*b3, a1*b2-a2*b1]
cross_product([A1, A2, A3], [B1, B2, B3], [C1, C2, C3]) :- C1 is A2 * B3 - A3 * B2, C2 is A3 * B1 - A1 * B3, C3 is A1 * B2 - A2 * B1.

% Matrix-vector multiplication: [[a,b,c],[d,e,f]] * [x,y,z] = [ax+by+cz, dx+ey+fz]
matrix_vector_multiply(Matrix, Vector, Result) :- maplist(dot_product(Vector), Matrix, Result).

% Vector-matrix multiplication: [x,y] * [[a,b,c],[d,e,f]] = [xa+yd, xb+ye, xc+yf]
vector_matrix_multiply(Vector, Matrix, Result) :- transpose(Matrix, TransposedMatrix), maplist(dot_product(Vector), TransposedMatrix, Result).

% Matrix multiplication: A * B = C where C[i][j] = sum(A[i][k] * B[k][j]) 
matrix_multiply(A, B, C) :- transpose(B, BT), maplist([Row,ResultRow]>>maplist(dot_product(Row), BT, ResultRow), A, C).

% Dot product 
dot_product(A, B, Result) :- maplist([X,Y,Z]>>(Z is X * Y), A, B, Products), sum_list(Products, Result).

% Transpose helper method
transpose([], []).
transpose([[]|_], []) :- !.
transpose(Matrix, [Col|Cols]) :- maplist([Row,Head]>>(Row = [Head|_]), Matrix, Col), maplist([Row,Tail]>>(Row = [_|Tail]), Matrix, Rest), transpose(Rest, Cols).

% Stack manipulation operators
stack_operator("DROP", [_|Rest], Rest).
stack_operator("DROP", [], []).
stack_operator("DUP", [H|T], [H, H|T]).
stack_operator("DUP", [], []).
stack_operator("SWAP", [A, B|Rest], [B, A|Rest]).
stack_operator("SWAP", Stack, Stack).
stack_operator("ROT", [A, B, C|Rest], [C, A, B|Rest]).
stack_operator("ROT", Stack, Stack).

% ROLL and ROLLD operation
stack_operator(Op, [N|Stack], New) :-
    member(Op, ["ROLL", "ROLLD"]), integer(N), N > 0, length(Stack, Len),
    (N =< Len ->split_at(N, Stack, Items, Rest), 
        (Op = "ROLL" ->  append(Init, [Last], Items), append([Last|Init], Rest, New)
        ; Items = [Head|Tail], append(Tail, [Head], Moved), append(Moved, Rest, New))
    ; New = [N|Stack]).

% Helper predicate to split list
split_at(0, List, [], List) :- !.
split_at(N, [H|T], [H|Start], End) :- N > 0, Next is N - 1, split_at(Next, T, Start, End).
split_at(_, [], [], []).

% Comparison operation
comparison_operator(Op, [B, A|Rest], [Result|Rest]) :-
    (Op = "==" -> (A == B -> Result = true ; Result = false)
    ; Op = "!=" -> (A \== B -> Result = true ; Result = false)
    ; Op = ">" -> ((number(A), number(B)) -> (A > B -> Result = true ; Result = false) ; (A @> B -> Result = true ; Result = false))
    ; Op = "<" ->  ((number(A), number(B)) -> (A < B -> Result = true ; Result = false) ; (A @< B -> Result = true ; Result = false))
    ; Op = ">=" ->  ((number(A), number(B)) -> (A >= B -> Result = true ; Result = false) ; (A @>= B -> Result = true ; Result = false))
    ; Op = "<=" ->  ((number(A), number(B)) -> (A =< B -> Result = true ; Result = false) ; (A @=< B -> Result = true ; Result = false))
    ; Op = "<=>" -> spaceship_comparison(A, B, Result)
    ; Result = false).
comparison_operator(_, Stack, Stack).

% Spaceship operator for comparison operation
spaceship_comparison(A, B, Result) :- ((number(A), number(B)) ->  (A < B -> Result = -1 ; A > B -> Result = 1 ; Result = 0) ; (A @< B -> Result = -1 ;   A @> B -> Result = 1 ;   Result = 0)).

% Boolean operation
boolean_operator(Op, [B, A|Rest], NewStack) :- (Op = "^" ->  xor_operation(A, B, Result), NewStack = [Result|Rest] ; boolean_operation(A, B, Op, Rest, NewStack)).
boolean_operator(_, Stack, Stack).
boolean_operation(A, B, Op, Rest, NewStack) :-
    ((A = true; A = false), (B = true; B = false) 
        -> (Op = "&" -> (A = true, B = true -> Result = true ; Result = false) ; Op = "|" -> ((A = true ; B = true) -> Result = true ; Result = false) ; Result = false) , NewStack = [Result|Rest]
    ; NewStack = [B, A|Rest]).

% XOR operation
xor_operation(A, B, Result) :-
    ((A = true; A = false), (B = true; B = false) -> (A \= B -> Result = true ; Result = false)
    ; (integer(A), integer(B)) -> Result is A xor B
    ; Result = error).

% Bitwise operation
bitwise_operator(Op, [B, A|Rest], [Result|Rest]) :-
    ((integer(A), integer(B)) ->  (Op = "<<" -> Result is A << B ; Op = ">>" -> Result is A >> B ; Result = error) ; Result = error).
bitwise_operator(_, Stack, Stack).

% Unary operation
unary_operator(Op, [A|Rest], [Result|Rest]) :-
    (Op = "!", (A = true; A = false) -> (A = true -> Result = false ; Result = true) 
    ; Op = "~", integer(A) ->  Result is \A
    ; Result = error).
unary_operator(_, Stack, Stack).

% Control operation
ifelse_operator("IFELSE", [Condition, FalseVal, TrueVal|Rest], [Result|Rest]) :- ((Condition = true; Condition = false) -> (Condition = true -> Result = TrueVal ; Result = FalseVal) ; fail).
ifelse_operator(_, Stack, Stack).

% Transpose operation
transpose_operator([Matrix|Rest], [Result|Rest]) :-
    (\+ (is_list(Matrix), Matrix = [Row|_], is_list(Row)) -> Result = Matrix  % check non-matrix
    ; Matrix = [] -> Result = []
    ; Matrix = [[]|_] -> Result = []
    ; Matrix = [[H|T]|Rows] -> maplist([R,Head]>>(R = [Head|_]), [[H|T]|Rows], Col), maplist([R,Tail]>>(R = [_|Tail]), [[H|T]|Rows], Tails), transpose_operator([Tails], [More]), Result = [Col|More]
    ; Result = Matrix).  % fallback
transpose_operator(Stack, Stack).

% EVAL operation
eval_operator([Top|Rest], Result) :-
    (string(Top) ->   % Convert the top element to tokens and evaluate
        % Check if it a lambda. If it a lambda, extract the content and apply it, else tokenise and evaluate regular string
        (sub_string(Top, 0, 1, _, "{"), sub_string(Top, _, 1, 0, "}") -> sub_string(Top, 1, _, 1, Content), lambda_operator(Content, Rest, Result) 
        ; process_token(Top, Tokens), evaluate_stack(Tokens, Rest, Result))
    ; atom(Top) -> atom_string(Top, Str), process_token(Str, Tokens), evaluate_stack(Tokens, Rest, Result)
    ; Result = [Top|Rest]).  % leave it on the stack
eval_operator(Stack, Stack).

% Lambda operation {n | x1, x0 ...}
lambda_operator(Perm, Stack, New) :-
    % Parse the permutation string and tokenise operations
    split_string(Perm, "|", " ", [NStr, RestStr]), number_string(N, NStr), process_token(RestStr, Tokens), separate_indices(Tokens, IdxTokens, OpTokens),
    maplist([token(literal, Str), Index]>>(atom_string(Atom, Str), atom_concat('x', NumAtom, Atom), atom_number(NumAtom, Index)), IdxTokens, Indices),  
    (length(Stack, Len), N =< Len ->    % Extract top N elements and change according to indices
        split_at(N, Stack, Top, Rest), reverse(Top, Rev),    % Reverse the args
        maplist([Index,Element]>>(nth0(Index, Top, Element) -> true ; Element = 0), Indices, Items),
        string_concat("{", Perm, Temp), string_concat(Temp, "}", Full), % Create full string for SELF references
        execute_lambda(OpTokens, Items, Full, Rev, Final), % Evaluate operations with lambda elements
        append(Final, Rest, New)
    ; New = Stack ).  % Not enough elements
   
% Handles SELF and index references for lambda operator 
execute_lambda([], Stack, _, _, Stack).
execute_lambda([token(literal, "SELF")|Tokens], Stack, Perm, Refs, Result) :- !, execute_lambda(Tokens, [Perm|Stack], Perm, Refs, Result).  % Push the lambda element onto the stack as a literal
execute_lambda([Token|Tokens], Stack, Perm, Refs, Result) :-
    (Token = token(literal, Str), atom_string(Atom, Str), atom_concat('x', NumAtom, Atom), atom_number(NumAtom, Index) ->   % Check if this is an index reference
        (nth0(Index, Refs, Value) ->  New = [Value|Stack] ; New = Stack)    % If it is an index reference, push the value from the original position, else don't push anything  
    ; evaluate_stack([Token], Stack, New)), execute_lambda(Tokens, New, Perm, Refs, Result).    % Regular evaluation for other tokens

% Separate tokens that are indices (x0, x1) for lambda operator and execute lambda
separate_indices([], [], []).
separate_indices([Token|Tokens], Indices, Ops) :-
    (Token = token(literal, Str), atom_string(Atom, Str), atom_concat('x', NumAtom, Atom), atom_number(NumAtom, _) -> Indices = [Token|Rest], separate_indices(Tokens, Rest, Ops)
    ; Indices = [], Ops = [Token|Tokens]).

% Write output
write_output(Stream, Stack) :- reverse(Stack, Rev), write_stack_elements(Stream, Rev).

% Write stack elements to output stream, one per line in recursion
write_stack_elements(_, []).
write_stack_elements(Stream, [Element|Rest]) :- format_element(Element, Fmt), format(Stream, '~w~n', [Fmt]), write_stack_elements(Stream, Rest).

% Format elements
format_element(Element, Fmt) :-
    (is_list(Element) -> format_array(Element, Fmt)
    ; string(Element) -> (is_operator(Element) ->  Fmt = Element ; atom_concat('"', Element, Temp), atom_concat(Temp, '"', Fmt))
    ; Fmt = Element).

% Format arrays
format_array([], '[]').
format_array([H|T], Fmt) :- maplist(format_element, [H|T], Items), atomic_list_concat(Items, ', ', Str), atom_concat('[', Str, Temp), atom_concat(Temp, ']', Fmt).