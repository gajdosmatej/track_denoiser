&
ХЈ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
С
	AvgPool3D

input"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"
Ttype:
2

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
ж
Conv3D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"!
	dilations	list(int)	

$
DisableCopyOnRead
resource
.
Identity

input"T
output"T"	
Ttype

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
А
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.15.02unknown8уё
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
Њ
Adam/v/conv3d_18/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/conv3d_18/bias/*
dtype0*
shape:*&
shared_nameAdam/v/conv3d_18/bias
{
)Adam/v/conv3d_18/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_18/bias*
_output_shapes
:*
dtype0
Њ
Adam/m/conv3d_18/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/conv3d_18/bias/*
dtype0*
shape:*&
shared_nameAdam/m/conv3d_18/bias
{
)Adam/m/conv3d_18/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_18/bias*
_output_shapes
:*
dtype0
Р
Adam/v/conv3d_18/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/conv3d_18/kernel/*
dtype0*
shape:*(
shared_nameAdam/v/conv3d_18/kernel

+Adam/v/conv3d_18/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_18/kernel**
_output_shapes
:*
dtype0
Р
Adam/m/conv3d_18/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/conv3d_18/kernel/*
dtype0*
shape:*(
shared_nameAdam/m/conv3d_18/kernel

+Adam/m/conv3d_18/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_18/kernel**
_output_shapes
:*
dtype0
Њ
Adam/v/conv3d_17/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/conv3d_17/bias/*
dtype0*
shape:*&
shared_nameAdam/v/conv3d_17/bias
{
)Adam/v/conv3d_17/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_17/bias*
_output_shapes
:*
dtype0
Њ
Adam/m/conv3d_17/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/conv3d_17/bias/*
dtype0*
shape:*&
shared_nameAdam/m/conv3d_17/bias
{
)Adam/m/conv3d_17/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_17/bias*
_output_shapes
:*
dtype0
Р
Adam/v/conv3d_17/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/conv3d_17/kernel/*
dtype0*
shape:*(
shared_nameAdam/v/conv3d_17/kernel

+Adam/v/conv3d_17/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_17/kernel**
_output_shapes
:*
dtype0
Р
Adam/m/conv3d_17/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/conv3d_17/kernel/*
dtype0*
shape:*(
shared_nameAdam/m/conv3d_17/kernel

+Adam/m/conv3d_17/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_17/kernel**
_output_shapes
:*
dtype0
Њ
Adam/v/conv3d_16/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/conv3d_16/bias/*
dtype0*
shape:*&
shared_nameAdam/v/conv3d_16/bias
{
)Adam/v/conv3d_16/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_16/bias*
_output_shapes
:*
dtype0
Њ
Adam/m/conv3d_16/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/conv3d_16/bias/*
dtype0*
shape:*&
shared_nameAdam/m/conv3d_16/bias
{
)Adam/m/conv3d_16/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_16/bias*
_output_shapes
:*
dtype0
Р
Adam/v/conv3d_16/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/conv3d_16/kernel/*
dtype0*
shape:*(
shared_nameAdam/v/conv3d_16/kernel

+Adam/v/conv3d_16/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_16/kernel**
_output_shapes
:*
dtype0
Р
Adam/m/conv3d_16/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/conv3d_16/kernel/*
dtype0*
shape:*(
shared_nameAdam/m/conv3d_16/kernel

+Adam/m/conv3d_16/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_16/kernel**
_output_shapes
:*
dtype0
Њ
Adam/v/conv3d_15/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/conv3d_15/bias/*
dtype0*
shape:*&
shared_nameAdam/v/conv3d_15/bias
{
)Adam/v/conv3d_15/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_15/bias*
_output_shapes
:*
dtype0
Њ
Adam/m/conv3d_15/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/conv3d_15/bias/*
dtype0*
shape:*&
shared_nameAdam/m/conv3d_15/bias
{
)Adam/m/conv3d_15/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_15/bias*
_output_shapes
:*
dtype0
Р
Adam/v/conv3d_15/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/conv3d_15/kernel/*
dtype0*
shape:*(
shared_nameAdam/v/conv3d_15/kernel

+Adam/v/conv3d_15/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_15/kernel**
_output_shapes
:*
dtype0
Р
Adam/m/conv3d_15/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/conv3d_15/kernel/*
dtype0*
shape:*(
shared_nameAdam/m/conv3d_15/kernel

+Adam/m/conv3d_15/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_15/kernel**
_output_shapes
:*
dtype0
Ї
Adam/v/conv3d_1/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/conv3d_1/bias/*
dtype0*
shape:*%
shared_nameAdam/v/conv3d_1/bias
y
(Adam/v/conv3d_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_1/bias*
_output_shapes
:*
dtype0
Ї
Adam/m/conv3d_1/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/conv3d_1/bias/*
dtype0*
shape:*%
shared_nameAdam/m/conv3d_1/bias
y
(Adam/m/conv3d_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_1/bias*
_output_shapes
:*
dtype0
Н
Adam/v/conv3d_1/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/conv3d_1/kernel/*
dtype0*
shape:*'
shared_nameAdam/v/conv3d_1/kernel

*Adam/v/conv3d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_1/kernel**
_output_shapes
:*
dtype0
Н
Adam/m/conv3d_1/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/conv3d_1/kernel/*
dtype0*
shape:*'
shared_nameAdam/m/conv3d_1/kernel

*Adam/m/conv3d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_1/kernel**
_output_shapes
:*
dtype0
Ё
Adam/v/conv3d/biasVarHandleOp*
_output_shapes
: *#

debug_nameAdam/v/conv3d/bias/*
dtype0*
shape:*#
shared_nameAdam/v/conv3d/bias
u
&Adam/v/conv3d/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d/bias*
_output_shapes
:*
dtype0
Ё
Adam/m/conv3d/biasVarHandleOp*
_output_shapes
: *#

debug_nameAdam/m/conv3d/bias/*
dtype0*
shape:*#
shared_nameAdam/m/conv3d/bias
u
&Adam/m/conv3d/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d/bias*
_output_shapes
:*
dtype0
З
Adam/v/conv3d/kernelVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/conv3d/kernel/*
dtype0*
shape:*%
shared_nameAdam/v/conv3d/kernel

(Adam/v/conv3d/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d/kernel**
_output_shapes
:*
dtype0
З
Adam/m/conv3d/kernelVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/conv3d/kernel/*
dtype0*
shape:*%
shared_nameAdam/m/conv3d/kernel

(Adam/m/conv3d/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d/kernel**
_output_shapes
:*
dtype0
Њ
Adam/v/conv3d_14/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/conv3d_14/bias/*
dtype0*
shape:*&
shared_nameAdam/v/conv3d_14/bias
{
)Adam/v/conv3d_14/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_14/bias*
_output_shapes
:*
dtype0
Њ
Adam/m/conv3d_14/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/conv3d_14/bias/*
dtype0*
shape:*&
shared_nameAdam/m/conv3d_14/bias
{
)Adam/m/conv3d_14/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_14/bias*
_output_shapes
:*
dtype0
Р
Adam/v/conv3d_14/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/conv3d_14/kernel/*
dtype0*
shape:*(
shared_nameAdam/v/conv3d_14/kernel

+Adam/v/conv3d_14/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_14/kernel**
_output_shapes
:*
dtype0
Р
Adam/m/conv3d_14/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/conv3d_14/kernel/*
dtype0*
shape:*(
shared_nameAdam/m/conv3d_14/kernel

+Adam/m/conv3d_14/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_14/kernel**
_output_shapes
:*
dtype0
Њ
Adam/v/conv3d_13/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/conv3d_13/bias/*
dtype0*
shape:*&
shared_nameAdam/v/conv3d_13/bias
{
)Adam/v/conv3d_13/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_13/bias*
_output_shapes
:*
dtype0
Њ
Adam/m/conv3d_13/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/conv3d_13/bias/*
dtype0*
shape:*&
shared_nameAdam/m/conv3d_13/bias
{
)Adam/m/conv3d_13/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_13/bias*
_output_shapes
:*
dtype0
Р
Adam/v/conv3d_13/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/conv3d_13/kernel/*
dtype0*
shape:*(
shared_nameAdam/v/conv3d_13/kernel

+Adam/v/conv3d_13/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_13/kernel**
_output_shapes
:*
dtype0
Р
Adam/m/conv3d_13/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/conv3d_13/kernel/*
dtype0*
shape:*(
shared_nameAdam/m/conv3d_13/kernel

+Adam/m/conv3d_13/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_13/kernel**
_output_shapes
:*
dtype0
Ї
Adam/v/conv3d_3/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/conv3d_3/bias/*
dtype0*
shape:*%
shared_nameAdam/v/conv3d_3/bias
y
(Adam/v/conv3d_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_3/bias*
_output_shapes
:*
dtype0
Ї
Adam/m/conv3d_3/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/conv3d_3/bias/*
dtype0*
shape:*%
shared_nameAdam/m/conv3d_3/bias
y
(Adam/m/conv3d_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_3/bias*
_output_shapes
:*
dtype0
Н
Adam/v/conv3d_3/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/conv3d_3/kernel/*
dtype0*
shape:*'
shared_nameAdam/v/conv3d_3/kernel

*Adam/v/conv3d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_3/kernel**
_output_shapes
:*
dtype0
Н
Adam/m/conv3d_3/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/conv3d_3/kernel/*
dtype0*
shape:*'
shared_nameAdam/m/conv3d_3/kernel

*Adam/m/conv3d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_3/kernel**
_output_shapes
:*
dtype0
Ї
Adam/v/conv3d_2/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/conv3d_2/bias/*
dtype0*
shape:*%
shared_nameAdam/v/conv3d_2/bias
y
(Adam/v/conv3d_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_2/bias*
_output_shapes
:*
dtype0
Ї
Adam/m/conv3d_2/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/conv3d_2/bias/*
dtype0*
shape:*%
shared_nameAdam/m/conv3d_2/bias
y
(Adam/m/conv3d_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_2/bias*
_output_shapes
:*
dtype0
Н
Adam/v/conv3d_2/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/conv3d_2/kernel/*
dtype0*
shape:*'
shared_nameAdam/v/conv3d_2/kernel

*Adam/v/conv3d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_2/kernel**
_output_shapes
:*
dtype0
Н
Adam/m/conv3d_2/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/conv3d_2/kernel/*
dtype0*
shape:*'
shared_nameAdam/m/conv3d_2/kernel

*Adam/m/conv3d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_2/kernel**
_output_shapes
:*
dtype0
Њ
Adam/v/conv3d_12/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/conv3d_12/bias/*
dtype0*
shape:*&
shared_nameAdam/v/conv3d_12/bias
{
)Adam/v/conv3d_12/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_12/bias*
_output_shapes
:*
dtype0
Њ
Adam/m/conv3d_12/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/conv3d_12/bias/*
dtype0*
shape:*&
shared_nameAdam/m/conv3d_12/bias
{
)Adam/m/conv3d_12/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_12/bias*
_output_shapes
:*
dtype0
Р
Adam/v/conv3d_12/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/conv3d_12/kernel/*
dtype0*
shape:*(
shared_nameAdam/v/conv3d_12/kernel

+Adam/v/conv3d_12/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_12/kernel**
_output_shapes
:*
dtype0
Р
Adam/m/conv3d_12/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/conv3d_12/kernel/*
dtype0*
shape:*(
shared_nameAdam/m/conv3d_12/kernel

+Adam/m/conv3d_12/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_12/kernel**
_output_shapes
:*
dtype0
Њ
Adam/v/conv3d_11/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/conv3d_11/bias/*
dtype0*
shape:*&
shared_nameAdam/v/conv3d_11/bias
{
)Adam/v/conv3d_11/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_11/bias*
_output_shapes
:*
dtype0
Њ
Adam/m/conv3d_11/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/conv3d_11/bias/*
dtype0*
shape:*&
shared_nameAdam/m/conv3d_11/bias
{
)Adam/m/conv3d_11/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_11/bias*
_output_shapes
:*
dtype0
Р
Adam/v/conv3d_11/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/conv3d_11/kernel/*
dtype0*
shape:*(
shared_nameAdam/v/conv3d_11/kernel

+Adam/v/conv3d_11/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_11/kernel**
_output_shapes
:*
dtype0
Р
Adam/m/conv3d_11/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/conv3d_11/kernel/*
dtype0*
shape:*(
shared_nameAdam/m/conv3d_11/kernel

+Adam/m/conv3d_11/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_11/kernel**
_output_shapes
:*
dtype0
Ї
Adam/v/conv3d_5/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/conv3d_5/bias/*
dtype0*
shape:*%
shared_nameAdam/v/conv3d_5/bias
y
(Adam/v/conv3d_5/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_5/bias*
_output_shapes
:*
dtype0
Ї
Adam/m/conv3d_5/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/conv3d_5/bias/*
dtype0*
shape:*%
shared_nameAdam/m/conv3d_5/bias
y
(Adam/m/conv3d_5/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_5/bias*
_output_shapes
:*
dtype0
Н
Adam/v/conv3d_5/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/conv3d_5/kernel/*
dtype0*
shape:*'
shared_nameAdam/v/conv3d_5/kernel

*Adam/v/conv3d_5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_5/kernel**
_output_shapes
:*
dtype0
Н
Adam/m/conv3d_5/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/conv3d_5/kernel/*
dtype0*
shape:*'
shared_nameAdam/m/conv3d_5/kernel

*Adam/m/conv3d_5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_5/kernel**
_output_shapes
:*
dtype0
Ї
Adam/v/conv3d_4/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/conv3d_4/bias/*
dtype0*
shape:*%
shared_nameAdam/v/conv3d_4/bias
y
(Adam/v/conv3d_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_4/bias*
_output_shapes
:*
dtype0
Ї
Adam/m/conv3d_4/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/conv3d_4/bias/*
dtype0*
shape:*%
shared_nameAdam/m/conv3d_4/bias
y
(Adam/m/conv3d_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_4/bias*
_output_shapes
:*
dtype0
Н
Adam/v/conv3d_4/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/conv3d_4/kernel/*
dtype0*
shape:*'
shared_nameAdam/v/conv3d_4/kernel

*Adam/v/conv3d_4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_4/kernel**
_output_shapes
:*
dtype0
Н
Adam/m/conv3d_4/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/conv3d_4/kernel/*
dtype0*
shape:*'
shared_nameAdam/m/conv3d_4/kernel

*Adam/m/conv3d_4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_4/kernel**
_output_shapes
:*
dtype0
Њ
Adam/v/conv3d_10/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/conv3d_10/bias/*
dtype0*
shape:*&
shared_nameAdam/v/conv3d_10/bias
{
)Adam/v/conv3d_10/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_10/bias*
_output_shapes
:*
dtype0
Њ
Adam/m/conv3d_10/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/conv3d_10/bias/*
dtype0*
shape:*&
shared_nameAdam/m/conv3d_10/bias
{
)Adam/m/conv3d_10/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_10/bias*
_output_shapes
:*
dtype0
Р
Adam/v/conv3d_10/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/conv3d_10/kernel/*
dtype0*
shape:*(
shared_nameAdam/v/conv3d_10/kernel

+Adam/v/conv3d_10/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_10/kernel**
_output_shapes
:*
dtype0
Р
Adam/m/conv3d_10/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/conv3d_10/kernel/*
dtype0*
shape:*(
shared_nameAdam/m/conv3d_10/kernel

+Adam/m/conv3d_10/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_10/kernel**
_output_shapes
:*
dtype0
Ї
Adam/v/conv3d_9/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/conv3d_9/bias/*
dtype0*
shape:*%
shared_nameAdam/v/conv3d_9/bias
y
(Adam/v/conv3d_9/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_9/bias*
_output_shapes
:*
dtype0
Ї
Adam/m/conv3d_9/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/conv3d_9/bias/*
dtype0*
shape:*%
shared_nameAdam/m/conv3d_9/bias
y
(Adam/m/conv3d_9/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_9/bias*
_output_shapes
:*
dtype0
Н
Adam/v/conv3d_9/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/conv3d_9/kernel/*
dtype0*
shape:*'
shared_nameAdam/v/conv3d_9/kernel

*Adam/v/conv3d_9/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_9/kernel**
_output_shapes
:*
dtype0
Н
Adam/m/conv3d_9/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/conv3d_9/kernel/*
dtype0*
shape:*'
shared_nameAdam/m/conv3d_9/kernel

*Adam/m/conv3d_9/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_9/kernel**
_output_shapes
:*
dtype0
Ї
Adam/v/conv3d_8/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/conv3d_8/bias/*
dtype0*
shape:*%
shared_nameAdam/v/conv3d_8/bias
y
(Adam/v/conv3d_8/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_8/bias*
_output_shapes
:*
dtype0
Ї
Adam/m/conv3d_8/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/conv3d_8/bias/*
dtype0*
shape:*%
shared_nameAdam/m/conv3d_8/bias
y
(Adam/m/conv3d_8/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_8/bias*
_output_shapes
:*
dtype0
Н
Adam/v/conv3d_8/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/conv3d_8/kernel/*
dtype0*
shape:*'
shared_nameAdam/v/conv3d_8/kernel

*Adam/v/conv3d_8/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_8/kernel**
_output_shapes
:*
dtype0
Н
Adam/m/conv3d_8/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/conv3d_8/kernel/*
dtype0*
shape:*'
shared_nameAdam/m/conv3d_8/kernel

*Adam/m/conv3d_8/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_8/kernel**
_output_shapes
:*
dtype0
Ї
Adam/v/conv3d_7/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/conv3d_7/bias/*
dtype0*
shape:*%
shared_nameAdam/v/conv3d_7/bias
y
(Adam/v/conv3d_7/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_7/bias*
_output_shapes
:*
dtype0
Ї
Adam/m/conv3d_7/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/conv3d_7/bias/*
dtype0*
shape:*%
shared_nameAdam/m/conv3d_7/bias
y
(Adam/m/conv3d_7/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_7/bias*
_output_shapes
:*
dtype0
Н
Adam/v/conv3d_7/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/conv3d_7/kernel/*
dtype0*
shape:*'
shared_nameAdam/v/conv3d_7/kernel

*Adam/v/conv3d_7/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_7/kernel**
_output_shapes
:*
dtype0
Н
Adam/m/conv3d_7/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/conv3d_7/kernel/*
dtype0*
shape:*'
shared_nameAdam/m/conv3d_7/kernel

*Adam/m/conv3d_7/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_7/kernel**
_output_shapes
:*
dtype0
Ї
Adam/v/conv3d_6/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/conv3d_6/bias/*
dtype0*
shape:*%
shared_nameAdam/v/conv3d_6/bias
y
(Adam/v/conv3d_6/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_6/bias*
_output_shapes
:*
dtype0
Ї
Adam/m/conv3d_6/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/conv3d_6/bias/*
dtype0*
shape:*%
shared_nameAdam/m/conv3d_6/bias
y
(Adam/m/conv3d_6/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_6/bias*
_output_shapes
:*
dtype0
Н
Adam/v/conv3d_6/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/conv3d_6/kernel/*
dtype0*
shape:*'
shared_nameAdam/v/conv3d_6/kernel

*Adam/v/conv3d_6/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_6/kernel**
_output_shapes
:*
dtype0
Н
Adam/m/conv3d_6/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/conv3d_6/kernel/*
dtype0*
shape:*'
shared_nameAdam/m/conv3d_6/kernel

*Adam/m/conv3d_6/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_6/kernel**
_output_shapes
:*
dtype0

learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0

	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	

conv3d_18/biasVarHandleOp*
_output_shapes
: *

debug_nameconv3d_18/bias/*
dtype0*
shape:*
shared_nameconv3d_18/bias
m
"conv3d_18/bias/Read/ReadVariableOpReadVariableOpconv3d_18/bias*
_output_shapes
:*
dtype0
Ћ
conv3d_18/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv3d_18/kernel/*
dtype0*
shape:*!
shared_nameconv3d_18/kernel

$conv3d_18/kernel/Read/ReadVariableOpReadVariableOpconv3d_18/kernel**
_output_shapes
:*
dtype0

conv3d_17/biasVarHandleOp*
_output_shapes
: *

debug_nameconv3d_17/bias/*
dtype0*
shape:*
shared_nameconv3d_17/bias
m
"conv3d_17/bias/Read/ReadVariableOpReadVariableOpconv3d_17/bias*
_output_shapes
:*
dtype0
Ћ
conv3d_17/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv3d_17/kernel/*
dtype0*
shape:*!
shared_nameconv3d_17/kernel

$conv3d_17/kernel/Read/ReadVariableOpReadVariableOpconv3d_17/kernel**
_output_shapes
:*
dtype0

conv3d_16/biasVarHandleOp*
_output_shapes
: *

debug_nameconv3d_16/bias/*
dtype0*
shape:*
shared_nameconv3d_16/bias
m
"conv3d_16/bias/Read/ReadVariableOpReadVariableOpconv3d_16/bias*
_output_shapes
:*
dtype0
Ћ
conv3d_16/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv3d_16/kernel/*
dtype0*
shape:*!
shared_nameconv3d_16/kernel

$conv3d_16/kernel/Read/ReadVariableOpReadVariableOpconv3d_16/kernel**
_output_shapes
:*
dtype0

conv3d_15/biasVarHandleOp*
_output_shapes
: *

debug_nameconv3d_15/bias/*
dtype0*
shape:*
shared_nameconv3d_15/bias
m
"conv3d_15/bias/Read/ReadVariableOpReadVariableOpconv3d_15/bias*
_output_shapes
:*
dtype0
Ћ
conv3d_15/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv3d_15/kernel/*
dtype0*
shape:*!
shared_nameconv3d_15/kernel

$conv3d_15/kernel/Read/ReadVariableOpReadVariableOpconv3d_15/kernel**
_output_shapes
:*
dtype0

conv3d_1/biasVarHandleOp*
_output_shapes
: *

debug_nameconv3d_1/bias/*
dtype0*
shape:*
shared_nameconv3d_1/bias
k
!conv3d_1/bias/Read/ReadVariableOpReadVariableOpconv3d_1/bias*
_output_shapes
:*
dtype0
Ј
conv3d_1/kernelVarHandleOp*
_output_shapes
: * 

debug_nameconv3d_1/kernel/*
dtype0*
shape:* 
shared_nameconv3d_1/kernel

#conv3d_1/kernel/Read/ReadVariableOpReadVariableOpconv3d_1/kernel**
_output_shapes
:*
dtype0

conv3d/biasVarHandleOp*
_output_shapes
: *

debug_nameconv3d/bias/*
dtype0*
shape:*
shared_nameconv3d/bias
g
conv3d/bias/Read/ReadVariableOpReadVariableOpconv3d/bias*
_output_shapes
:*
dtype0
Ђ
conv3d/kernelVarHandleOp*
_output_shapes
: *

debug_nameconv3d/kernel/*
dtype0*
shape:*
shared_nameconv3d/kernel
{
!conv3d/kernel/Read/ReadVariableOpReadVariableOpconv3d/kernel**
_output_shapes
:*
dtype0

conv3d_14/biasVarHandleOp*
_output_shapes
: *

debug_nameconv3d_14/bias/*
dtype0*
shape:*
shared_nameconv3d_14/bias
m
"conv3d_14/bias/Read/ReadVariableOpReadVariableOpconv3d_14/bias*
_output_shapes
:*
dtype0
Ћ
conv3d_14/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv3d_14/kernel/*
dtype0*
shape:*!
shared_nameconv3d_14/kernel

$conv3d_14/kernel/Read/ReadVariableOpReadVariableOpconv3d_14/kernel**
_output_shapes
:*
dtype0

conv3d_13/biasVarHandleOp*
_output_shapes
: *

debug_nameconv3d_13/bias/*
dtype0*
shape:*
shared_nameconv3d_13/bias
m
"conv3d_13/bias/Read/ReadVariableOpReadVariableOpconv3d_13/bias*
_output_shapes
:*
dtype0
Ћ
conv3d_13/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv3d_13/kernel/*
dtype0*
shape:*!
shared_nameconv3d_13/kernel

$conv3d_13/kernel/Read/ReadVariableOpReadVariableOpconv3d_13/kernel**
_output_shapes
:*
dtype0

conv3d_3/biasVarHandleOp*
_output_shapes
: *

debug_nameconv3d_3/bias/*
dtype0*
shape:*
shared_nameconv3d_3/bias
k
!conv3d_3/bias/Read/ReadVariableOpReadVariableOpconv3d_3/bias*
_output_shapes
:*
dtype0
Ј
conv3d_3/kernelVarHandleOp*
_output_shapes
: * 

debug_nameconv3d_3/kernel/*
dtype0*
shape:* 
shared_nameconv3d_3/kernel

#conv3d_3/kernel/Read/ReadVariableOpReadVariableOpconv3d_3/kernel**
_output_shapes
:*
dtype0

conv3d_2/biasVarHandleOp*
_output_shapes
: *

debug_nameconv3d_2/bias/*
dtype0*
shape:*
shared_nameconv3d_2/bias
k
!conv3d_2/bias/Read/ReadVariableOpReadVariableOpconv3d_2/bias*
_output_shapes
:*
dtype0
Ј
conv3d_2/kernelVarHandleOp*
_output_shapes
: * 

debug_nameconv3d_2/kernel/*
dtype0*
shape:* 
shared_nameconv3d_2/kernel

#conv3d_2/kernel/Read/ReadVariableOpReadVariableOpconv3d_2/kernel**
_output_shapes
:*
dtype0

conv3d_12/biasVarHandleOp*
_output_shapes
: *

debug_nameconv3d_12/bias/*
dtype0*
shape:*
shared_nameconv3d_12/bias
m
"conv3d_12/bias/Read/ReadVariableOpReadVariableOpconv3d_12/bias*
_output_shapes
:*
dtype0
Ћ
conv3d_12/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv3d_12/kernel/*
dtype0*
shape:*!
shared_nameconv3d_12/kernel

$conv3d_12/kernel/Read/ReadVariableOpReadVariableOpconv3d_12/kernel**
_output_shapes
:*
dtype0

conv3d_11/biasVarHandleOp*
_output_shapes
: *

debug_nameconv3d_11/bias/*
dtype0*
shape:*
shared_nameconv3d_11/bias
m
"conv3d_11/bias/Read/ReadVariableOpReadVariableOpconv3d_11/bias*
_output_shapes
:*
dtype0
Ћ
conv3d_11/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv3d_11/kernel/*
dtype0*
shape:*!
shared_nameconv3d_11/kernel

$conv3d_11/kernel/Read/ReadVariableOpReadVariableOpconv3d_11/kernel**
_output_shapes
:*
dtype0

conv3d_5/biasVarHandleOp*
_output_shapes
: *

debug_nameconv3d_5/bias/*
dtype0*
shape:*
shared_nameconv3d_5/bias
k
!conv3d_5/bias/Read/ReadVariableOpReadVariableOpconv3d_5/bias*
_output_shapes
:*
dtype0
Ј
conv3d_5/kernelVarHandleOp*
_output_shapes
: * 

debug_nameconv3d_5/kernel/*
dtype0*
shape:* 
shared_nameconv3d_5/kernel

#conv3d_5/kernel/Read/ReadVariableOpReadVariableOpconv3d_5/kernel**
_output_shapes
:*
dtype0

conv3d_4/biasVarHandleOp*
_output_shapes
: *

debug_nameconv3d_4/bias/*
dtype0*
shape:*
shared_nameconv3d_4/bias
k
!conv3d_4/bias/Read/ReadVariableOpReadVariableOpconv3d_4/bias*
_output_shapes
:*
dtype0
Ј
conv3d_4/kernelVarHandleOp*
_output_shapes
: * 

debug_nameconv3d_4/kernel/*
dtype0*
shape:* 
shared_nameconv3d_4/kernel

#conv3d_4/kernel/Read/ReadVariableOpReadVariableOpconv3d_4/kernel**
_output_shapes
:*
dtype0

conv3d_10/biasVarHandleOp*
_output_shapes
: *

debug_nameconv3d_10/bias/*
dtype0*
shape:*
shared_nameconv3d_10/bias
m
"conv3d_10/bias/Read/ReadVariableOpReadVariableOpconv3d_10/bias*
_output_shapes
:*
dtype0
Ћ
conv3d_10/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv3d_10/kernel/*
dtype0*
shape:*!
shared_nameconv3d_10/kernel

$conv3d_10/kernel/Read/ReadVariableOpReadVariableOpconv3d_10/kernel**
_output_shapes
:*
dtype0

conv3d_9/biasVarHandleOp*
_output_shapes
: *

debug_nameconv3d_9/bias/*
dtype0*
shape:*
shared_nameconv3d_9/bias
k
!conv3d_9/bias/Read/ReadVariableOpReadVariableOpconv3d_9/bias*
_output_shapes
:*
dtype0
Ј
conv3d_9/kernelVarHandleOp*
_output_shapes
: * 

debug_nameconv3d_9/kernel/*
dtype0*
shape:* 
shared_nameconv3d_9/kernel

#conv3d_9/kernel/Read/ReadVariableOpReadVariableOpconv3d_9/kernel**
_output_shapes
:*
dtype0

conv3d_8/biasVarHandleOp*
_output_shapes
: *

debug_nameconv3d_8/bias/*
dtype0*
shape:*
shared_nameconv3d_8/bias
k
!conv3d_8/bias/Read/ReadVariableOpReadVariableOpconv3d_8/bias*
_output_shapes
:*
dtype0
Ј
conv3d_8/kernelVarHandleOp*
_output_shapes
: * 

debug_nameconv3d_8/kernel/*
dtype0*
shape:* 
shared_nameconv3d_8/kernel

#conv3d_8/kernel/Read/ReadVariableOpReadVariableOpconv3d_8/kernel**
_output_shapes
:*
dtype0

conv3d_7/biasVarHandleOp*
_output_shapes
: *

debug_nameconv3d_7/bias/*
dtype0*
shape:*
shared_nameconv3d_7/bias
k
!conv3d_7/bias/Read/ReadVariableOpReadVariableOpconv3d_7/bias*
_output_shapes
:*
dtype0
Ј
conv3d_7/kernelVarHandleOp*
_output_shapes
: * 

debug_nameconv3d_7/kernel/*
dtype0*
shape:* 
shared_nameconv3d_7/kernel

#conv3d_7/kernel/Read/ReadVariableOpReadVariableOpconv3d_7/kernel**
_output_shapes
:*
dtype0

conv3d_6/biasVarHandleOp*
_output_shapes
: *

debug_nameconv3d_6/bias/*
dtype0*
shape:*
shared_nameconv3d_6/bias
k
!conv3d_6/bias/Read/ReadVariableOpReadVariableOpconv3d_6/bias*
_output_shapes
:*
dtype0
Ј
conv3d_6/kernelVarHandleOp*
_output_shapes
: * 

debug_nameconv3d_6/kernel/*
dtype0*
shape:* 
shared_nameconv3d_6/kernel

#conv3d_6/kernel/Read/ReadVariableOpReadVariableOpconv3d_6/kernel**
_output_shapes
:*
dtype0

serving_default_input_1Placeholder*4
_output_shapes"
 :џџџџџџџџџа*
dtype0*)
shape :џџџџџџџџџа
ћ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv3d_6/kernelconv3d_6/biasconv3d_7/kernelconv3d_7/biasconv3d_8/kernelconv3d_8/biasconv3d_9/kernelconv3d_9/biasconv3d_4/kernelconv3d_4/biasconv3d_10/kernelconv3d_10/biasconv3d_5/kernelconv3d_5/biasconv3d_11/kernelconv3d_11/biasconv3d_2/kernelconv3d_2/biasconv3d_12/kernelconv3d_12/biasconv3d_3/kernelconv3d_3/biasconv3d_13/kernelconv3d_13/biasconv3d/kernelconv3d/biasconv3d_14/kernelconv3d_14/biasconv3d_1/kernelconv3d_1/biasconv3d_15/kernelconv3d_15/biasconv3d_16/kernelconv3d_16/biasconv3d_17/kernelconv3d_17/biasconv3d_18/kernelconv3d_18/bias*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_17189

NoOpNoOp
ё
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ћ
value B B
ы
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer_with_weights-8
layer-16
layer_with_weights-9
layer-17
layer-18
layer_with_weights-10
layer-19
layer-20
layer_with_weights-11
layer-21
layer_with_weights-12
layer-22
layer_with_weights-13
layer-23
layer-24
layer_with_weights-14
layer-25
layer-26
layer_with_weights-15
layer-27
layer_with_weights-16
layer-28
layer_with_weights-17
layer-29
layer_with_weights-18
layer-30
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses
&_default_save_signature
'	optimizer
(
signatures*
* 
Ш
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias
 1_jit_compiled_convolution_op*

2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses* 
Ш
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias
 @_jit_compiled_convolution_op*

A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses* 
Ш
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses

Mkernel
Nbias
 O_jit_compiled_convolution_op*

P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses* 

V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses* 
Ш
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

bkernel
cbias
 d_jit_compiled_convolution_op*

e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses* 
Ш
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

qkernel
rbias
 s_jit_compiled_convolution_op*
Ш
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses

zkernel
{bias
 |_jit_compiled_convolution_op*

}	variables
~trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
б
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
б
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op*
б
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses
Ёkernel
	Ђbias
!Ѓ_jit_compiled_convolution_op*
б
Є	variables
Ѕtrainable_variables
Іregularization_losses
Ї	keras_api
Ј__call__
+Љ&call_and_return_all_conditional_losses
Њkernel
	Ћbias
!Ќ_jit_compiled_convolution_op*

­	variables
Ўtrainable_variables
Џregularization_losses
А	keras_api
Б__call__
+В&call_and_return_all_conditional_losses* 
б
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
З__call__
+И&call_and_return_all_conditional_losses
Йkernel
	Кbias
!Л_jit_compiled_convolution_op*

М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses* 
б
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses
Шkernel
	Щbias
!Ъ_jit_compiled_convolution_op*
б
Ы	variables
Ьtrainable_variables
Эregularization_losses
Ю	keras_api
Я__call__
+а&call_and_return_all_conditional_losses
бkernel
	вbias
!г_jit_compiled_convolution_op*
б
д	variables
еtrainable_variables
жregularization_losses
з	keras_api
и__call__
+й&call_and_return_all_conditional_losses
кkernel
	лbias
!м_jit_compiled_convolution_op*

н	variables
оtrainable_variables
пregularization_losses
р	keras_api
с__call__
+т&call_and_return_all_conditional_losses* 
б
у	variables
фtrainable_variables
хregularization_losses
ц	keras_api
ч__call__
+ш&call_and_return_all_conditional_losses
щkernel
	ъbias
!ы_jit_compiled_convolution_op*

ь	variables
эtrainable_variables
юregularization_losses
я	keras_api
№__call__
+ё&call_and_return_all_conditional_losses* 
б
ђ	variables
ѓtrainable_variables
єregularization_losses
ѕ	keras_api
і__call__
+ї&call_and_return_all_conditional_losses
јkernel
	љbias
!њ_jit_compiled_convolution_op*
б
ћ	variables
ќtrainable_variables
§regularization_losses
ў	keras_api
џ__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op*
б
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op*
б
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op*
Ф
/0
01
>2
?3
M4
N5
b6
c7
q8
r9
z10
{11
12
13
14
15
Ё16
Ђ17
Њ18
Ћ19
Й20
К21
Ш22
Щ23
б24
в25
к26
л27
щ28
ъ29
ј30
љ31
32
33
34
35
36
37*
Ф
/0
01
>2
?3
M4
N5
b6
c7
q8
r9
z10
{11
12
13
14
15
Ё16
Ђ17
Њ18
Ћ19
Й20
К21
Ш22
Щ23
б24
в25
к26
л27
щ28
ъ29
ј30
љ31
32
33
34
35
36
37*
* 
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
&_default_save_signature
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 


_variables
 _iterations
Ё_learning_rate
Ђ_index_dict
Ѓ
_momentums
Є_velocities
Ѕ_update_step_xla*

Іserving_default* 

/0
01*

/0
01*
* 

Їnon_trainable_variables
Јlayers
Љmetrics
 Њlayer_regularization_losses
Ћlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

Ќtrace_0* 

­trace_0* 
_Y
VARIABLE_VALUEconv3d_6/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_6/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Ўnon_trainable_variables
Џlayers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses* 

Гtrace_0* 

Дtrace_0* 

>0
?1*

>0
?1*
* 

Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*

Кtrace_0* 

Лtrace_0* 
_Y
VARIABLE_VALUEconv3d_7/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_7/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses* 

Сtrace_0* 

Тtrace_0* 

M0
N1*

M0
N1*
* 

Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*

Шtrace_0* 

Щtrace_0* 
_Y
VARIABLE_VALUEconv3d_8/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_8/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Ъnon_trainable_variables
Ыlayers
Ьmetrics
 Эlayer_regularization_losses
Юlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses* 

Яtrace_0* 

аtrace_0* 
* 
* 
* 

бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses* 

жtrace_0* 

зtrace_0* 

b0
c1*

b0
c1*
* 

иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*

нtrace_0* 

оtrace_0* 
_Y
VARIABLE_VALUEconv3d_9/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_9/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

пnon_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses* 

фtrace_0* 

хtrace_0* 

q0
r1*

q0
r1*
* 

цnon_trainable_variables
чlayers
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses*

ыtrace_0* 

ьtrace_0* 
`Z
VARIABLE_VALUEconv3d_10/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv3d_10/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

z0
{1*

z0
{1*
* 

эnon_trainable_variables
юlayers
яmetrics
 №layer_regularization_losses
ёlayer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses*

ђtrace_0* 

ѓtrace_0* 
_Y
VARIABLE_VALUEconv3d_4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_4/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

єnon_trainable_variables
ѕlayers
іmetrics
 їlayer_regularization_losses
јlayer_metrics
}	variables
~trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

љtrace_0* 

њtrace_0* 

0
1*

0
1*
* 

ћnon_trainable_variables
ќlayers
§metrics
 ўlayer_regularization_losses
џlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
_Y
VARIABLE_VALUEconv3d_5/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_5/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEconv3d_11/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv3d_11/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

Ё0
Ђ1*

Ё0
Ђ1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEconv3d_12/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv3d_12/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

Њ0
Ћ1*

Њ0
Ћ1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Є	variables
Ѕtrainable_variables
Іregularization_losses
Ј__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses*

trace_0* 

trace_0* 
_Y
VARIABLE_VALUEconv3d_2/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_2/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
­	variables
Ўtrainable_variables
Џregularization_losses
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses* 

Ѓtrace_0* 

Єtrace_0* 

Й0
К1*

Й0
К1*
* 

Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
Г	variables
Дtrainable_variables
Еregularization_losses
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses*

Њtrace_0* 

Ћtrace_0* 
`Z
VARIABLE_VALUEconv3d_3/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv3d_3/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Ќnon_trainable_variables
­layers
Ўmetrics
 Џlayer_regularization_losses
Аlayer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses* 

Бtrace_0* 

Вtrace_0* 

Ш0
Щ1*

Ш0
Щ1*
* 

Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
Т	variables
Уtrainable_variables
Фregularization_losses
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses*

Иtrace_0* 

Йtrace_0* 
a[
VARIABLE_VALUEconv3d_13/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv3d_13/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

б0
в1*

б0
в1*
* 

Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
Ы	variables
Ьtrainable_variables
Эregularization_losses
Я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses*

Пtrace_0* 

Рtrace_0* 
a[
VARIABLE_VALUEconv3d_14/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv3d_14/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

к0
л1*

к0
л1*
* 

Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
д	variables
еtrainable_variables
жregularization_losses
и__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses*

Цtrace_0* 

Чtrace_0* 
^X
VARIABLE_VALUEconv3d/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv3d/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
н	variables
оtrainable_variables
пregularization_losses
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses* 

Эtrace_0* 

Юtrace_0* 

щ0
ъ1*

щ0
ъ1*
* 

Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
у	variables
фtrainable_variables
хregularization_losses
ч__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses*

дtrace_0* 

еtrace_0* 
`Z
VARIABLE_VALUEconv3d_1/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv3d_1/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
ь	variables
эtrainable_variables
юregularization_losses
№__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses* 

лtrace_0* 

мtrace_0* 

ј0
љ1*

ј0
љ1*
* 

нnon_trainable_variables
оlayers
пmetrics
 рlayer_regularization_losses
сlayer_metrics
ђ	variables
ѓtrainable_variables
єregularization_losses
і__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses*

тtrace_0* 

уtrace_0* 
a[
VARIABLE_VALUEconv3d_15/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv3d_15/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

0
1*
* 

фnon_trainable_variables
хlayers
цmetrics
 чlayer_regularization_losses
шlayer_metrics
ћ	variables
ќtrainable_variables
§regularization_losses
џ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

щtrace_0* 

ъtrace_0* 
a[
VARIABLE_VALUEconv3d_16/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv3d_16/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

0
1*
* 

ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

№trace_0* 

ёtrace_0* 
a[
VARIABLE_VALUEconv3d_17/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv3d_17/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

0
1*
* 

ђnon_trainable_variables
ѓlayers
єmetrics
 ѕlayer_regularization_losses
іlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

їtrace_0* 

јtrace_0* 
a[
VARIABLE_VALUEconv3d_18/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv3d_18/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
ђ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30*

љ0*
* 
* 
* 
* 
* 
* 
Џ
 0
њ1
ћ2
ќ3
§4
ў5
џ6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
 39
Ё40
Ђ41
Ѓ42
Є43
Ѕ44
І45
Ї46
Ј47
Љ48
Њ49
Ћ50
Ќ51
­52
Ў53
Џ54
А55
Б56
В57
Г58
Д59
Е60
Ж61
З62
И63
Й64
К65
Л66
М67
Н68
О69
П70
Р71
С72
Т73
У74
Ф75
Х76*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
а
њ0
ќ1
ў2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19
Ђ20
Є21
І22
Ј23
Њ24
Ќ25
Ў26
А27
В28
Д29
Ж30
И31
К32
М33
О34
Р35
Т36
Ф37*
а
ћ0
§1
џ2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
Ё19
Ѓ20
Ѕ21
Ї22
Љ23
Ћ24
­25
Џ26
Б27
Г28
Е29
З30
Й31
Л32
Н33
П34
С35
У36
Х37*
В
Цtrace_0
Чtrace_1
Шtrace_2
Щtrace_3
Ъtrace_4
Ыtrace_5
Ьtrace_6
Эtrace_7
Юtrace_8
Яtrace_9
аtrace_10
бtrace_11
вtrace_12
гtrace_13
дtrace_14
еtrace_15
жtrace_16
зtrace_17
иtrace_18
йtrace_19
кtrace_20
лtrace_21
мtrace_22
нtrace_23
оtrace_24
пtrace_25
рtrace_26
сtrace_27
тtrace_28
уtrace_29
фtrace_30
хtrace_31
цtrace_32
чtrace_33
шtrace_34
щtrace_35
ъtrace_36
ыtrace_37* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
ь	variables
э	keras_api

юtotal

яcount*
a[
VARIABLE_VALUEAdam/m/conv3d_6/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv3d_6/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/conv3d_6/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv3d_6/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv3d_7/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv3d_7/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/conv3d_7/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv3d_7/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv3d_8/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv3d_8/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv3d_8/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv3d_8/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv3d_9/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv3d_9/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv3d_9/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv3d_9/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv3d_10/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv3d_10/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv3d_10/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv3d_10/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv3d_4/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv3d_4/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv3d_4/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv3d_4/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv3d_5/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv3d_5/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv3d_5/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv3d_5/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv3d_11/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv3d_11/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv3d_11/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv3d_11/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv3d_12/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv3d_12/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv3d_12/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv3d_12/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv3d_2/kernel2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv3d_2/kernel2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv3d_2/bias2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv3d_2/bias2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv3d_3/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv3d_3/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv3d_3/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv3d_3/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv3d_13/kernel2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv3d_13/kernel2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv3d_13/bias2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv3d_13/bias2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv3d_14/kernel2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv3d_14/kernel2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv3d_14/bias2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv3d_14/bias2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv3d/kernel2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv3d/kernel2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/conv3d/bias2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/conv3d/bias2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv3d_1/kernel2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv3d_1/kernel2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv3d_1/bias2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv3d_1/bias2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv3d_15/kernel2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv3d_15/kernel2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv3d_15/bias2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv3d_15/bias2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv3d_16/kernel2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv3d_16/kernel2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv3d_16/bias2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv3d_16/bias2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv3d_17/kernel2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv3d_17/kernel2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv3d_17/bias2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv3d_17/bias2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv3d_18/kernel2optimizer/_variables/73/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv3d_18/kernel2optimizer/_variables/74/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv3d_18/bias2optimizer/_variables/75/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv3d_18/bias2optimizer/_variables/76/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

ю0
я1*

ь	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv3d_6/kernelconv3d_6/biasconv3d_7/kernelconv3d_7/biasconv3d_8/kernelconv3d_8/biasconv3d_9/kernelconv3d_9/biasconv3d_10/kernelconv3d_10/biasconv3d_4/kernelconv3d_4/biasconv3d_5/kernelconv3d_5/biasconv3d_11/kernelconv3d_11/biasconv3d_12/kernelconv3d_12/biasconv3d_2/kernelconv3d_2/biasconv3d_3/kernelconv3d_3/biasconv3d_13/kernelconv3d_13/biasconv3d_14/kernelconv3d_14/biasconv3d/kernelconv3d/biasconv3d_1/kernelconv3d_1/biasconv3d_15/kernelconv3d_15/biasconv3d_16/kernelconv3d_16/biasconv3d_17/kernelconv3d_17/biasconv3d_18/kernelconv3d_18/bias	iterationlearning_rateAdam/m/conv3d_6/kernelAdam/v/conv3d_6/kernelAdam/m/conv3d_6/biasAdam/v/conv3d_6/biasAdam/m/conv3d_7/kernelAdam/v/conv3d_7/kernelAdam/m/conv3d_7/biasAdam/v/conv3d_7/biasAdam/m/conv3d_8/kernelAdam/v/conv3d_8/kernelAdam/m/conv3d_8/biasAdam/v/conv3d_8/biasAdam/m/conv3d_9/kernelAdam/v/conv3d_9/kernelAdam/m/conv3d_9/biasAdam/v/conv3d_9/biasAdam/m/conv3d_10/kernelAdam/v/conv3d_10/kernelAdam/m/conv3d_10/biasAdam/v/conv3d_10/biasAdam/m/conv3d_4/kernelAdam/v/conv3d_4/kernelAdam/m/conv3d_4/biasAdam/v/conv3d_4/biasAdam/m/conv3d_5/kernelAdam/v/conv3d_5/kernelAdam/m/conv3d_5/biasAdam/v/conv3d_5/biasAdam/m/conv3d_11/kernelAdam/v/conv3d_11/kernelAdam/m/conv3d_11/biasAdam/v/conv3d_11/biasAdam/m/conv3d_12/kernelAdam/v/conv3d_12/kernelAdam/m/conv3d_12/biasAdam/v/conv3d_12/biasAdam/m/conv3d_2/kernelAdam/v/conv3d_2/kernelAdam/m/conv3d_2/biasAdam/v/conv3d_2/biasAdam/m/conv3d_3/kernelAdam/v/conv3d_3/kernelAdam/m/conv3d_3/biasAdam/v/conv3d_3/biasAdam/m/conv3d_13/kernelAdam/v/conv3d_13/kernelAdam/m/conv3d_13/biasAdam/v/conv3d_13/biasAdam/m/conv3d_14/kernelAdam/v/conv3d_14/kernelAdam/m/conv3d_14/biasAdam/v/conv3d_14/biasAdam/m/conv3d/kernelAdam/v/conv3d/kernelAdam/m/conv3d/biasAdam/v/conv3d/biasAdam/m/conv3d_1/kernelAdam/v/conv3d_1/kernelAdam/m/conv3d_1/biasAdam/v/conv3d_1/biasAdam/m/conv3d_15/kernelAdam/v/conv3d_15/kernelAdam/m/conv3d_15/biasAdam/v/conv3d_15/biasAdam/m/conv3d_16/kernelAdam/v/conv3d_16/kernelAdam/m/conv3d_16/biasAdam/v/conv3d_16/biasAdam/m/conv3d_17/kernelAdam/v/conv3d_17/kernelAdam/m/conv3d_17/biasAdam/v/conv3d_17/biasAdam/m/conv3d_18/kernelAdam/v/conv3d_18/kernelAdam/m/conv3d_18/biasAdam/v/conv3d_18/biastotalcountConst*
Tin|
z2x*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_18892

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d_6/kernelconv3d_6/biasconv3d_7/kernelconv3d_7/biasconv3d_8/kernelconv3d_8/biasconv3d_9/kernelconv3d_9/biasconv3d_10/kernelconv3d_10/biasconv3d_4/kernelconv3d_4/biasconv3d_5/kernelconv3d_5/biasconv3d_11/kernelconv3d_11/biasconv3d_12/kernelconv3d_12/biasconv3d_2/kernelconv3d_2/biasconv3d_3/kernelconv3d_3/biasconv3d_13/kernelconv3d_13/biasconv3d_14/kernelconv3d_14/biasconv3d/kernelconv3d/biasconv3d_1/kernelconv3d_1/biasconv3d_15/kernelconv3d_15/biasconv3d_16/kernelconv3d_16/biasconv3d_17/kernelconv3d_17/biasconv3d_18/kernelconv3d_18/bias	iterationlearning_rateAdam/m/conv3d_6/kernelAdam/v/conv3d_6/kernelAdam/m/conv3d_6/biasAdam/v/conv3d_6/biasAdam/m/conv3d_7/kernelAdam/v/conv3d_7/kernelAdam/m/conv3d_7/biasAdam/v/conv3d_7/biasAdam/m/conv3d_8/kernelAdam/v/conv3d_8/kernelAdam/m/conv3d_8/biasAdam/v/conv3d_8/biasAdam/m/conv3d_9/kernelAdam/v/conv3d_9/kernelAdam/m/conv3d_9/biasAdam/v/conv3d_9/biasAdam/m/conv3d_10/kernelAdam/v/conv3d_10/kernelAdam/m/conv3d_10/biasAdam/v/conv3d_10/biasAdam/m/conv3d_4/kernelAdam/v/conv3d_4/kernelAdam/m/conv3d_4/biasAdam/v/conv3d_4/biasAdam/m/conv3d_5/kernelAdam/v/conv3d_5/kernelAdam/m/conv3d_5/biasAdam/v/conv3d_5/biasAdam/m/conv3d_11/kernelAdam/v/conv3d_11/kernelAdam/m/conv3d_11/biasAdam/v/conv3d_11/biasAdam/m/conv3d_12/kernelAdam/v/conv3d_12/kernelAdam/m/conv3d_12/biasAdam/v/conv3d_12/biasAdam/m/conv3d_2/kernelAdam/v/conv3d_2/kernelAdam/m/conv3d_2/biasAdam/v/conv3d_2/biasAdam/m/conv3d_3/kernelAdam/v/conv3d_3/kernelAdam/m/conv3d_3/biasAdam/v/conv3d_3/biasAdam/m/conv3d_13/kernelAdam/v/conv3d_13/kernelAdam/m/conv3d_13/biasAdam/v/conv3d_13/biasAdam/m/conv3d_14/kernelAdam/v/conv3d_14/kernelAdam/m/conv3d_14/biasAdam/v/conv3d_14/biasAdam/m/conv3d/kernelAdam/v/conv3d/kernelAdam/m/conv3d/biasAdam/v/conv3d/biasAdam/m/conv3d_1/kernelAdam/v/conv3d_1/kernelAdam/m/conv3d_1/biasAdam/v/conv3d_1/biasAdam/m/conv3d_15/kernelAdam/v/conv3d_15/kernelAdam/m/conv3d_15/biasAdam/v/conv3d_15/biasAdam/m/conv3d_16/kernelAdam/v/conv3d_16/kernelAdam/m/conv3d_16/biasAdam/v/conv3d_16/biasAdam/m/conv3d_17/kernelAdam/v/conv3d_17/kernelAdam/m/conv3d_17/biasAdam/v/conv3d_17/biasAdam/m/conv3d_18/kernelAdam/v/conv3d_18/kernelAdam/m/conv3d_18/biasAdam/v/conv3d_18/biastotalcount*
Tin{
y2w*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_19255
ж

D__inference_conv3d_13_layer_call_and_return_conditional_losses_16358

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџhm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџhS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџh
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_17329
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
­
Ё
(__inference_conv3d_1_layer_call_fn_18058

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_16550|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџа<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџа: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name18054:%!

_user_specified_name18052:\ X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs

r
F__inference_concatenate_layer_call_and_return_conditional_losses_17652
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџ4c
IdentityIdentityconcat:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:џџџџџџџџџ4:џџџџџџџџџ4:]Y
3
_output_shapes!
:џџџџџџџџџ4
"
_user_specified_name
inputs_1:] Y
3
_output_shapes!
:џџџџџџџџџ4
"
_user_specified_name
inputs_0

t
H__inference_concatenate_2_layer_call_and_return_conditional_losses_18082
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*4
_output_shapes"
 :џџџџџџџџџаd
IdentityIdentityconcat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџа"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџа:џџџџџџџџџа:^Z
4
_output_shapes"
 :џџџџџџџџџа
"
_user_specified_name
inputs_1:^ Z
4
_output_shapes"
 :џџџџџџџџџа
"
_user_specified_name
inputs_0
Ћ
J
"__inference__update_step_xla_17299
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
ж

D__inference_conv3d_13_layer_call_and_return_conditional_losses_17861

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџhm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџhS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџh
 
_user_specified_nameinputs
л
Z
"__inference__update_step_xla_17244
gradient&
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*+
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:T P
*
_output_shapes
:
"
_user_specified_name
gradient
!
о

%__inference_model_layer_call_fn_16820
input_1%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:'
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:(

unknown_11:

unknown_12:(

unknown_13:

unknown_14:(

unknown_15:

unknown_16:(

unknown_17:

unknown_18:(

unknown_19:

unknown_20:(

unknown_21:

unknown_22:(

unknown_23:

unknown_24:(

unknown_25:

unknown_26:(

unknown_27:

unknown_28:(

unknown_29:

unknown_30:(

unknown_31:

unknown_32:(

unknown_33:

unknown_34:(

unknown_35:

unknown_36:
identityЂStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_16629|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџа<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:џџџџџџџџџа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%&!

_user_specified_name16816:%%!

_user_specified_name16814:%$!

_user_specified_name16812:%#!

_user_specified_name16810:%"!

_user_specified_name16808:%!!

_user_specified_name16806:% !

_user_specified_name16804:%!

_user_specified_name16802:%!

_user_specified_name16800:%!

_user_specified_name16798:%!

_user_specified_name16796:%!

_user_specified_name16794:%!

_user_specified_name16792:%!

_user_specified_name16790:%!

_user_specified_name16788:%!

_user_specified_name16786:%!

_user_specified_name16784:%!

_user_specified_name16782:%!

_user_specified_name16780:%!

_user_specified_name16778:%!

_user_specified_name16776:%!

_user_specified_name16774:%!

_user_specified_name16772:%!

_user_specified_name16770:%!

_user_specified_name16768:%!

_user_specified_name16766:%!

_user_specified_name16764:%!

_user_specified_name16762:%
!

_user_specified_name16760:%	!

_user_specified_name16758:%!

_user_specified_name16756:%!

_user_specified_name16754:%!

_user_specified_name16752:%!

_user_specified_name16750:%!

_user_specified_name16748:%!

_user_specified_name16746:%!

_user_specified_name16744:%!

_user_specified_name16742:] Y
4
_output_shapes"
 :џџџџџџџџџа
!
_user_specified_name	input_1
е

C__inference_conv3d_2_layer_call_and_return_conditional_losses_16210

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџhm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџhS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџh
 
_user_specified_nameinputs
м
j
N__inference_average_pooling3d_1_layer_call_and_return_conditional_losses_15975

inputs
identityО
	AvgPool3D	AvgPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingVALID*
strides	

IdentityIdentityAvgPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
е

C__inference_conv3d_8_layer_call_and_return_conditional_losses_17459

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ4m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ4S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџ4
 
_user_specified_nameinputs
м
j
N__inference_average_pooling3d_5_layer_call_and_return_conditional_losses_15965

inputs
identityО
	AvgPool3D	AvgPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingVALID*
strides	

IdentityIdentityAvgPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ж

D__inference_conv3d_14_layer_call_and_return_conditional_losses_17881

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџhm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџhS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџh
 
_user_specified_nameinputs
е

C__inference_conv3d_7_layer_call_and_return_conditional_losses_17429

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџhm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџhS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџh
 
_user_specified_nameinputs
м

D__inference_conv3d_16_layer_call_and_return_conditional_losses_18122

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџаn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџаS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџа: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs
Љ
Ё
(__inference_conv3d_2_layer_call_fn_17701

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_16210{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџh<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџh: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name17697:%!

_user_specified_name17695:[ W
3
_output_shapes!
:џџџџџџџџџh
 
_user_specified_nameinputs
л
Z
"__inference__update_step_xla_17284
gradient&
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*+
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:T P
*
_output_shapes
:
"
_user_specified_name
gradient
е

C__inference_conv3d_9_layer_call_and_return_conditional_losses_17499

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
ж

D__inference_conv3d_10_layer_call_and_return_conditional_losses_17529

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
й
ў
A__inference_conv3d_layer_call_and_return_conditional_losses_16374

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџаn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџаS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџа: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs
е

C__inference_conv3d_5_layer_call_and_return_conditional_losses_16170

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ4m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ4S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџ4
 
_user_specified_nameinputs
е

C__inference_conv3d_4_layer_call_and_return_conditional_losses_16072

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ4m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ4S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџ4
 
_user_specified_nameinputs

r
H__inference_concatenate_1_layer_call_and_return_conditional_losses_16346

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџhc
IdentityIdentityconcat:output:0*
T0*3
_output_shapes!
:џџџџџџџџџh"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:џџџџџџџџџh:џџџџџџџџџh:[W
3
_output_shapes!
:џџџџџџџџџh
 
_user_specified_nameinputs:[ W
3
_output_shapes!
:џџџџџџџџџh
 
_user_specified_nameinputs
Љ
Ё
(__inference_conv3d_9_layer_call_fn_17488

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_9_layer_call_and_return_conditional_losses_16056{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name17484:%!

_user_specified_name17482:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Џ
Ђ
)__inference_conv3d_17_layer_call_fn_18131

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_17_layer_call_and_return_conditional_losses_16606|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџа<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџа: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name18127:%!

_user_specified_name18125:\ X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs
л
Z
"__inference__update_step_xla_17354
gradient&
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*+
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:T P
*
_output_shapes
:
"
_user_specified_name
gradient
е

C__inference_conv3d_2_layer_call_and_return_conditional_losses_17712

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџhm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџhS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџh
 
_user_specified_nameinputs
л
Z
"__inference__update_step_xla_17214
gradient&
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*+
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:T P
*
_output_shapes
:
"
_user_specified_name
gradient
й
В
@__inference_model_layer_call_and_return_conditional_losses_16629
input_1,
conv3d_6_16004:
conv3d_6_16006:,
conv3d_7_16021:
conv3d_7_16023:,
conv3d_8_16038:
conv3d_8_16040:,
conv3d_9_16057:
conv3d_9_16059:,
conv3d_4_16073:
conv3d_4_16075:-
conv3d_10_16089:
conv3d_10_16091:,
conv3d_5_16171:
conv3d_5_16173:-
conv3d_11_16195:
conv3d_11_16197:,
conv3d_2_16211:
conv3d_2_16213:-
conv3d_12_16227:
conv3d_12_16229:,
conv3d_3_16335:
conv3d_3_16337:-
conv3d_13_16359:
conv3d_13_16361:*
conv3d_16375:
conv3d_16377:-
conv3d_14_16391:
conv3d_14_16393:,
conv3d_1_16551:
conv3d_1_16553:-
conv3d_15_16575:
conv3d_15_16577:-
conv3d_16_16591:
conv3d_16_16593:-
conv3d_17_16607:
conv3d_17_16609:-
conv3d_18_16623:
conv3d_18_16625:
identityЂconv3d/StatefulPartitionedCallЂ conv3d_1/StatefulPartitionedCallЂ!conv3d_10/StatefulPartitionedCallЂ!conv3d_11/StatefulPartitionedCallЂ!conv3d_12/StatefulPartitionedCallЂ!conv3d_13/StatefulPartitionedCallЂ!conv3d_14/StatefulPartitionedCallЂ!conv3d_15/StatefulPartitionedCallЂ!conv3d_16/StatefulPartitionedCallЂ!conv3d_17/StatefulPartitionedCallЂ!conv3d_18/StatefulPartitionedCallЂ conv3d_2/StatefulPartitionedCallЂ conv3d_3/StatefulPartitionedCallЂ conv3d_4/StatefulPartitionedCallЂ conv3d_5/StatefulPartitionedCallЂ conv3d_6/StatefulPartitionedCallЂ conv3d_7/StatefulPartitionedCallЂ conv3d_8/StatefulPartitionedCallЂ conv3d_9/StatefulPartitionedCallў
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_6_16004conv3d_6_16006*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_6_layer_call_and_return_conditional_losses_16003џ
#average_pooling3d_3/PartitionedCallPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling3d_3_layer_call_and_return_conditional_losses_15945Ђ
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling3d_3/PartitionedCall:output:0conv3d_7_16021conv3d_7_16023*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_7_layer_call_and_return_conditional_losses_16020џ
#average_pooling3d_4/PartitionedCallPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling3d_4_layer_call_and_return_conditional_losses_15955Ђ
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCall,average_pooling3d_4/PartitionedCall:output:0conv3d_8_16038conv3d_8_16040*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_8_layer_call_and_return_conditional_losses_16037н
#average_pooling3d_1/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling3d_1_layer_call_and_return_conditional_losses_15975џ
#average_pooling3d_5/PartitionedCallPartitionedCall)conv3d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling3d_5_layer_call_and_return_conditional_losses_15965
#average_pooling3d_2/PartitionedCallPartitionedCall,average_pooling3d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling3d_2_layer_call_and_return_conditional_losses_15985Ђ
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCall,average_pooling3d_5/PartitionedCall:output:0conv3d_9_16057conv3d_9_16059*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_9_layer_call_and_return_conditional_losses_16056Ђ
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall,average_pooling3d_2/PartitionedCall:output:0conv3d_4_16073conv3d_4_16075*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_16072Ѓ
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCall)conv3d_9/StatefulPartitionedCall:output:0conv3d_10_16089conv3d_10_16091*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_10_layer_call_and_return_conditional_losses_16088є
up_sampling3d/PartitionedCallPartitionedCall*conv3d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_up_sampling3d_layer_call_and_return_conditional_losses_16158
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0conv3d_5_16171conv3d_5_16173*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_16170
concatenate/PartitionedCallPartitionedCall&up_sampling3d/PartitionedCall:output:0)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_16182
!conv3d_11/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv3d_11_16195conv3d_11_16197*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_11_layer_call_and_return_conditional_losses_16194Ђ
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall,average_pooling3d_1/PartitionedCall:output:0conv3d_2_16211conv3d_2_16213*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_16210Є
!conv3d_12/StatefulPartitionedCallStatefulPartitionedCall*conv3d_11/StatefulPartitionedCall:output:0conv3d_12_16227conv3d_12_16229*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_12_layer_call_and_return_conditional_losses_16226ј
up_sampling3d_1/PartitionedCallPartitionedCall*conv3d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_16322
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0conv3d_3_16335conv3d_3_16337*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_16334
concatenate_1/PartitionedCallPartitionedCall(up_sampling3d_1/PartitionedCall:output:0)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_16346 
!conv3d_13/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv3d_13_16359conv3d_13_16361*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_13_layer_call_and_return_conditional_losses_16358і
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_16375conv3d_16377*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_16374Є
!conv3d_14/StatefulPartitionedCallStatefulPartitionedCall*conv3d_13/StatefulPartitionedCall:output:0conv3d_14_16391conv3d_14_16393*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_14_layer_call_and_return_conditional_losses_16390љ
up_sampling3d_2/PartitionedCallPartitionedCall*conv3d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_16538
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_16551conv3d_1_16553*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_16550
concatenate_2/PartitionedCallPartitionedCall(up_sampling3d_2/PartitionedCall:output:0)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_16562Ё
!conv3d_15/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv3d_15_16575conv3d_15_16577*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_15_layer_call_and_return_conditional_losses_16574Ѕ
!conv3d_16/StatefulPartitionedCallStatefulPartitionedCall*conv3d_15/StatefulPartitionedCall:output:0conv3d_16_16591conv3d_16_16593*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_16_layer_call_and_return_conditional_losses_16590Ѕ
!conv3d_17/StatefulPartitionedCallStatefulPartitionedCall*conv3d_16/StatefulPartitionedCall:output:0conv3d_17_16607conv3d_17_16609*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_17_layer_call_and_return_conditional_losses_16606Ѕ
!conv3d_18/StatefulPartitionedCallStatefulPartitionedCall*conv3d_17/StatefulPartitionedCall:output:0conv3d_18_16623conv3d_18_16625*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_18_layer_call_and_return_conditional_losses_16622
IdentityIdentity*conv3d_18/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџаТ
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall"^conv3d_11/StatefulPartitionedCall"^conv3d_12/StatefulPartitionedCall"^conv3d_13/StatefulPartitionedCall"^conv3d_14/StatefulPartitionedCall"^conv3d_15/StatefulPartitionedCall"^conv3d_16/StatefulPartitionedCall"^conv3d_17/StatefulPartitionedCall"^conv3d_18/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:џџџџџџџџџа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2F
!conv3d_10/StatefulPartitionedCall!conv3d_10/StatefulPartitionedCall2F
!conv3d_11/StatefulPartitionedCall!conv3d_11/StatefulPartitionedCall2F
!conv3d_12/StatefulPartitionedCall!conv3d_12/StatefulPartitionedCall2F
!conv3d_13/StatefulPartitionedCall!conv3d_13/StatefulPartitionedCall2F
!conv3d_14/StatefulPartitionedCall!conv3d_14/StatefulPartitionedCall2F
!conv3d_15/StatefulPartitionedCall!conv3d_15/StatefulPartitionedCall2F
!conv3d_16/StatefulPartitionedCall!conv3d_16/StatefulPartitionedCall2F
!conv3d_17/StatefulPartitionedCall!conv3d_17/StatefulPartitionedCall2F
!conv3d_18/StatefulPartitionedCall!conv3d_18/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2D
 conv3d_8/StatefulPartitionedCall conv3d_8/StatefulPartitionedCall2D
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall:%&!

_user_specified_name16625:%%!

_user_specified_name16623:%$!

_user_specified_name16609:%#!

_user_specified_name16607:%"!

_user_specified_name16593:%!!

_user_specified_name16591:% !

_user_specified_name16577:%!

_user_specified_name16575:%!

_user_specified_name16553:%!

_user_specified_name16551:%!

_user_specified_name16393:%!

_user_specified_name16391:%!

_user_specified_name16377:%!

_user_specified_name16375:%!

_user_specified_name16361:%!

_user_specified_name16359:%!

_user_specified_name16337:%!

_user_specified_name16335:%!

_user_specified_name16229:%!

_user_specified_name16227:%!

_user_specified_name16213:%!

_user_specified_name16211:%!

_user_specified_name16197:%!

_user_specified_name16195:%!

_user_specified_name16173:%!

_user_specified_name16171:%!

_user_specified_name16091:%!

_user_specified_name16089:%
!

_user_specified_name16075:%	!

_user_specified_name16073:%!

_user_specified_name16059:%!

_user_specified_name16057:%!

_user_specified_name16040:%!

_user_specified_name16038:%!

_user_specified_name16023:%!

_user_specified_name16021:%!

_user_specified_name16006:%!

_user_specified_name16004:] Y
4
_output_shapes"
 :џџџџџџџџџа
!
_user_specified_name	input_1
Ћ
J
"__inference__update_step_xla_17289
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
ж

D__inference_conv3d_11_layer_call_and_return_conditional_losses_17672

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ4m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ4S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџ4
 
_user_specified_nameinputs
е

C__inference_conv3d_8_layer_call_and_return_conditional_losses_16037

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ4m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ4S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџ4
 
_user_specified_nameinputs
м
j
N__inference_average_pooling3d_3_layer_call_and_return_conditional_losses_17409

inputs
identityО
	AvgPool3D	AvgPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingVALID*
strides	

IdentityIdentityAvgPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ц
ЋN
!__inference__traced_restore_19255
file_prefix>
 assignvariableop_conv3d_6_kernel:.
 assignvariableop_1_conv3d_6_bias:@
"assignvariableop_2_conv3d_7_kernel:.
 assignvariableop_3_conv3d_7_bias:@
"assignvariableop_4_conv3d_8_kernel:.
 assignvariableop_5_conv3d_8_bias:@
"assignvariableop_6_conv3d_9_kernel:.
 assignvariableop_7_conv3d_9_bias:A
#assignvariableop_8_conv3d_10_kernel:/
!assignvariableop_9_conv3d_10_bias:A
#assignvariableop_10_conv3d_4_kernel:/
!assignvariableop_11_conv3d_4_bias:A
#assignvariableop_12_conv3d_5_kernel:/
!assignvariableop_13_conv3d_5_bias:B
$assignvariableop_14_conv3d_11_kernel:0
"assignvariableop_15_conv3d_11_bias:B
$assignvariableop_16_conv3d_12_kernel:0
"assignvariableop_17_conv3d_12_bias:A
#assignvariableop_18_conv3d_2_kernel:/
!assignvariableop_19_conv3d_2_bias:A
#assignvariableop_20_conv3d_3_kernel:/
!assignvariableop_21_conv3d_3_bias:B
$assignvariableop_22_conv3d_13_kernel:0
"assignvariableop_23_conv3d_13_bias:B
$assignvariableop_24_conv3d_14_kernel:0
"assignvariableop_25_conv3d_14_bias:?
!assignvariableop_26_conv3d_kernel:-
assignvariableop_27_conv3d_bias:A
#assignvariableop_28_conv3d_1_kernel:/
!assignvariableop_29_conv3d_1_bias:B
$assignvariableop_30_conv3d_15_kernel:0
"assignvariableop_31_conv3d_15_bias:B
$assignvariableop_32_conv3d_16_kernel:0
"assignvariableop_33_conv3d_16_bias:B
$assignvariableop_34_conv3d_17_kernel:0
"assignvariableop_35_conv3d_17_bias:B
$assignvariableop_36_conv3d_18_kernel:0
"assignvariableop_37_conv3d_18_bias:'
assignvariableop_38_iteration:	 +
!assignvariableop_39_learning_rate: H
*assignvariableop_40_adam_m_conv3d_6_kernel:H
*assignvariableop_41_adam_v_conv3d_6_kernel:6
(assignvariableop_42_adam_m_conv3d_6_bias:6
(assignvariableop_43_adam_v_conv3d_6_bias:H
*assignvariableop_44_adam_m_conv3d_7_kernel:H
*assignvariableop_45_adam_v_conv3d_7_kernel:6
(assignvariableop_46_adam_m_conv3d_7_bias:6
(assignvariableop_47_adam_v_conv3d_7_bias:H
*assignvariableop_48_adam_m_conv3d_8_kernel:H
*assignvariableop_49_adam_v_conv3d_8_kernel:6
(assignvariableop_50_adam_m_conv3d_8_bias:6
(assignvariableop_51_adam_v_conv3d_8_bias:H
*assignvariableop_52_adam_m_conv3d_9_kernel:H
*assignvariableop_53_adam_v_conv3d_9_kernel:6
(assignvariableop_54_adam_m_conv3d_9_bias:6
(assignvariableop_55_adam_v_conv3d_9_bias:I
+assignvariableop_56_adam_m_conv3d_10_kernel:I
+assignvariableop_57_adam_v_conv3d_10_kernel:7
)assignvariableop_58_adam_m_conv3d_10_bias:7
)assignvariableop_59_adam_v_conv3d_10_bias:H
*assignvariableop_60_adam_m_conv3d_4_kernel:H
*assignvariableop_61_adam_v_conv3d_4_kernel:6
(assignvariableop_62_adam_m_conv3d_4_bias:6
(assignvariableop_63_adam_v_conv3d_4_bias:H
*assignvariableop_64_adam_m_conv3d_5_kernel:H
*assignvariableop_65_adam_v_conv3d_5_kernel:6
(assignvariableop_66_adam_m_conv3d_5_bias:6
(assignvariableop_67_adam_v_conv3d_5_bias:I
+assignvariableop_68_adam_m_conv3d_11_kernel:I
+assignvariableop_69_adam_v_conv3d_11_kernel:7
)assignvariableop_70_adam_m_conv3d_11_bias:7
)assignvariableop_71_adam_v_conv3d_11_bias:I
+assignvariableop_72_adam_m_conv3d_12_kernel:I
+assignvariableop_73_adam_v_conv3d_12_kernel:7
)assignvariableop_74_adam_m_conv3d_12_bias:7
)assignvariableop_75_adam_v_conv3d_12_bias:H
*assignvariableop_76_adam_m_conv3d_2_kernel:H
*assignvariableop_77_adam_v_conv3d_2_kernel:6
(assignvariableop_78_adam_m_conv3d_2_bias:6
(assignvariableop_79_adam_v_conv3d_2_bias:H
*assignvariableop_80_adam_m_conv3d_3_kernel:H
*assignvariableop_81_adam_v_conv3d_3_kernel:6
(assignvariableop_82_adam_m_conv3d_3_bias:6
(assignvariableop_83_adam_v_conv3d_3_bias:I
+assignvariableop_84_adam_m_conv3d_13_kernel:I
+assignvariableop_85_adam_v_conv3d_13_kernel:7
)assignvariableop_86_adam_m_conv3d_13_bias:7
)assignvariableop_87_adam_v_conv3d_13_bias:I
+assignvariableop_88_adam_m_conv3d_14_kernel:I
+assignvariableop_89_adam_v_conv3d_14_kernel:7
)assignvariableop_90_adam_m_conv3d_14_bias:7
)assignvariableop_91_adam_v_conv3d_14_bias:F
(assignvariableop_92_adam_m_conv3d_kernel:F
(assignvariableop_93_adam_v_conv3d_kernel:4
&assignvariableop_94_adam_m_conv3d_bias:4
&assignvariableop_95_adam_v_conv3d_bias:H
*assignvariableop_96_adam_m_conv3d_1_kernel:H
*assignvariableop_97_adam_v_conv3d_1_kernel:6
(assignvariableop_98_adam_m_conv3d_1_bias:6
(assignvariableop_99_adam_v_conv3d_1_bias:J
,assignvariableop_100_adam_m_conv3d_15_kernel:J
,assignvariableop_101_adam_v_conv3d_15_kernel:8
*assignvariableop_102_adam_m_conv3d_15_bias:8
*assignvariableop_103_adam_v_conv3d_15_bias:J
,assignvariableop_104_adam_m_conv3d_16_kernel:J
,assignvariableop_105_adam_v_conv3d_16_kernel:8
*assignvariableop_106_adam_m_conv3d_16_bias:8
*assignvariableop_107_adam_v_conv3d_16_bias:J
,assignvariableop_108_adam_m_conv3d_17_kernel:J
,assignvariableop_109_adam_v_conv3d_17_kernel:8
*assignvariableop_110_adam_m_conv3d_17_bias:8
*assignvariableop_111_adam_v_conv3d_17_bias:J
,assignvariableop_112_adam_m_conv3d_18_kernel:J
,assignvariableop_113_adam_v_conv3d_18_kernel:8
*assignvariableop_114_adam_m_conv3d_18_bias:8
*assignvariableop_115_adam_v_conv3d_18_bias:$
assignvariableop_116_total: $
assignvariableop_117_count: 
identity_119ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_100ЂAssignVariableOp_101ЂAssignVariableOp_102ЂAssignVariableOp_103ЂAssignVariableOp_104ЂAssignVariableOp_105ЂAssignVariableOp_106ЂAssignVariableOp_107ЂAssignVariableOp_108ЂAssignVariableOp_109ЂAssignVariableOp_11ЂAssignVariableOp_110ЂAssignVariableOp_111ЂAssignVariableOp_112ЂAssignVariableOp_113ЂAssignVariableOp_114ЂAssignVariableOp_115ЂAssignVariableOp_116ЂAssignVariableOp_117ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_58ЂAssignVariableOp_59ЂAssignVariableOp_6ЂAssignVariableOp_60ЂAssignVariableOp_61ЂAssignVariableOp_62ЂAssignVariableOp_63ЂAssignVariableOp_64ЂAssignVariableOp_65ЂAssignVariableOp_66ЂAssignVariableOp_67ЂAssignVariableOp_68ЂAssignVariableOp_69ЂAssignVariableOp_7ЂAssignVariableOp_70ЂAssignVariableOp_71ЂAssignVariableOp_72ЂAssignVariableOp_73ЂAssignVariableOp_74ЂAssignVariableOp_75ЂAssignVariableOp_76ЂAssignVariableOp_77ЂAssignVariableOp_78ЂAssignVariableOp_79ЂAssignVariableOp_8ЂAssignVariableOp_80ЂAssignVariableOp_81ЂAssignVariableOp_82ЂAssignVariableOp_83ЂAssignVariableOp_84ЂAssignVariableOp_85ЂAssignVariableOp_86ЂAssignVariableOp_87ЂAssignVariableOp_88ЂAssignVariableOp_89ЂAssignVariableOp_9ЂAssignVariableOp_90ЂAssignVariableOp_91ЂAssignVariableOp_92ЂAssignVariableOp_93ЂAssignVariableOp_94ЂAssignVariableOp_95ЂAssignVariableOp_96ЂAssignVariableOp_97ЂAssignVariableOp_98ЂAssignVariableOp_992
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:w*
dtype0*Љ1
value1B1wB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/73/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/74/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/75/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/76/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHс
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:w*
dtype0*
valueљBіwB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ѕ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ђ
_output_shapesп
м:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*
dtypes{
y2w	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOpAssignVariableOp assignvariableop_conv3d_6_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv3d_6_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv3d_7_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv3d_7_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv3d_8_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv3d_8_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv3d_9_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv3d_9_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv3d_10_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv3d_10_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv3d_4_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv3d_4_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv3d_5_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv3d_5_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv3d_11_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv3d_11_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_16AssignVariableOp$assignvariableop_16_conv3d_12_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_17AssignVariableOp"assignvariableop_17_conv3d_12_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv3d_2_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv3d_2_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_20AssignVariableOp#assignvariableop_20_conv3d_3_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_21AssignVariableOp!assignvariableop_21_conv3d_3_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_22AssignVariableOp$assignvariableop_22_conv3d_13_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_23AssignVariableOp"assignvariableop_23_conv3d_13_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv3d_14_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv3d_14_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_26AssignVariableOp!assignvariableop_26_conv3d_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_27AssignVariableOpassignvariableop_27_conv3d_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_28AssignVariableOp#assignvariableop_28_conv3d_1_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_29AssignVariableOp!assignvariableop_29_conv3d_1_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_30AssignVariableOp$assignvariableop_30_conv3d_15_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_31AssignVariableOp"assignvariableop_31_conv3d_15_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_32AssignVariableOp$assignvariableop_32_conv3d_16_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_33AssignVariableOp"assignvariableop_33_conv3d_16_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_34AssignVariableOp$assignvariableop_34_conv3d_17_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_35AssignVariableOp"assignvariableop_35_conv3d_17_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_36AssignVariableOp$assignvariableop_36_conv3d_18_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_37AssignVariableOp"assignvariableop_37_conv3d_18_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0	*
_output_shapes
:Ж
AssignVariableOp_38AssignVariableOpassignvariableop_38_iterationIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_39AssignVariableOp!assignvariableop_39_learning_rateIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_m_conv3d_6_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_v_conv3d_6_kernelIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_m_conv3d_6_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_v_conv3d_6_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_m_conv3d_7_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_v_conv3d_7_kernelIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_m_conv3d_7_biasIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_v_conv3d_7_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_m_conv3d_8_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_v_conv3d_8_kernelIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_m_conv3d_8_biasIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_v_conv3d_8_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_52AssignVariableOp*assignvariableop_52_adam_m_conv3d_9_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_v_conv3d_9_kernelIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_m_conv3d_9_biasIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_55AssignVariableOp(assignvariableop_55_adam_v_conv3d_9_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_56AssignVariableOp+assignvariableop_56_adam_m_conv3d_10_kernelIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_v_conv3d_10_kernelIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_m_conv3d_10_biasIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_59AssignVariableOp)assignvariableop_59_adam_v_conv3d_10_biasIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_60AssignVariableOp*assignvariableop_60_adam_m_conv3d_4_kernelIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_v_conv3d_4_kernelIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_m_conv3d_4_biasIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_63AssignVariableOp(assignvariableop_63_adam_v_conv3d_4_biasIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_64AssignVariableOp*assignvariableop_64_adam_m_conv3d_5_kernelIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_v_conv3d_5_kernelIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_m_conv3d_5_biasIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_67AssignVariableOp(assignvariableop_67_adam_v_conv3d_5_biasIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_68AssignVariableOp+assignvariableop_68_adam_m_conv3d_11_kernelIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_v_conv3d_11_kernelIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_m_conv3d_11_biasIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_71AssignVariableOp)assignvariableop_71_adam_v_conv3d_11_biasIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_72AssignVariableOp+assignvariableop_72_adam_m_conv3d_12_kernelIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_v_conv3d_12_kernelIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_m_conv3d_12_biasIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_75AssignVariableOp)assignvariableop_75_adam_v_conv3d_12_biasIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_76AssignVariableOp*assignvariableop_76_adam_m_conv3d_2_kernelIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_v_conv3d_2_kernelIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_m_conv3d_2_biasIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_79AssignVariableOp(assignvariableop_79_adam_v_conv3d_2_biasIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_80AssignVariableOp*assignvariableop_80_adam_m_conv3d_3_kernelIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_v_conv3d_3_kernelIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_m_conv3d_3_biasIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_83AssignVariableOp(assignvariableop_83_adam_v_conv3d_3_biasIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_84AssignVariableOp+assignvariableop_84_adam_m_conv3d_13_kernelIdentity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_85AssignVariableOp+assignvariableop_85_adam_v_conv3d_13_kernelIdentity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_86AssignVariableOp)assignvariableop_86_adam_m_conv3d_13_biasIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_87AssignVariableOp)assignvariableop_87_adam_v_conv3d_13_biasIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_88AssignVariableOp+assignvariableop_88_adam_m_conv3d_14_kernelIdentity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_89AssignVariableOp+assignvariableop_89_adam_v_conv3d_14_kernelIdentity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_90AssignVariableOp)assignvariableop_90_adam_m_conv3d_14_biasIdentity_90:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_91AssignVariableOp)assignvariableop_91_adam_v_conv3d_14_biasIdentity_91:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_92AssignVariableOp(assignvariableop_92_adam_m_conv3d_kernelIdentity_92:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_93AssignVariableOp(assignvariableop_93_adam_v_conv3d_kernelIdentity_93:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_94AssignVariableOp&assignvariableop_94_adam_m_conv3d_biasIdentity_94:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_95AssignVariableOp&assignvariableop_95_adam_v_conv3d_biasIdentity_95:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_96AssignVariableOp*assignvariableop_96_adam_m_conv3d_1_kernelIdentity_96:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_97AssignVariableOp*assignvariableop_97_adam_v_conv3d_1_kernelIdentity_97:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_98AssignVariableOp(assignvariableop_98_adam_m_conv3d_1_biasIdentity_98:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_99AssignVariableOp(assignvariableop_99_adam_v_conv3d_1_biasIdentity_99:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_100AssignVariableOp,assignvariableop_100_adam_m_conv3d_15_kernelIdentity_100:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_101AssignVariableOp,assignvariableop_101_adam_v_conv3d_15_kernelIdentity_101:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_102AssignVariableOp*assignvariableop_102_adam_m_conv3d_15_biasIdentity_102:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_103AssignVariableOp*assignvariableop_103_adam_v_conv3d_15_biasIdentity_103:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_104AssignVariableOp,assignvariableop_104_adam_m_conv3d_16_kernelIdentity_104:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_105AssignVariableOp,assignvariableop_105_adam_v_conv3d_16_kernelIdentity_105:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_106AssignVariableOp*assignvariableop_106_adam_m_conv3d_16_biasIdentity_106:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_107AssignVariableOp*assignvariableop_107_adam_v_conv3d_16_biasIdentity_107:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_108AssignVariableOp,assignvariableop_108_adam_m_conv3d_17_kernelIdentity_108:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_109AssignVariableOp,assignvariableop_109_adam_v_conv3d_17_kernelIdentity_109:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_110AssignVariableOp*assignvariableop_110_adam_m_conv3d_17_biasIdentity_110:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_111AssignVariableOp*assignvariableop_111_adam_v_conv3d_17_biasIdentity_111:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_112AssignVariableOp,assignvariableop_112_adam_m_conv3d_18_kernelIdentity_112:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_113AssignVariableOp,assignvariableop_113_adam_v_conv3d_18_kernelIdentity_113:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_114AssignVariableOp*assignvariableop_114_adam_m_conv3d_18_biasIdentity_114:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_115AssignVariableOp*assignvariableop_115_adam_v_conv3d_18_biasIdentity_115:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_116AssignVariableOpassignvariableop_116_totalIdentity_116:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_117AssignVariableOpassignvariableop_117_countIdentity_117:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 
Identity_118Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_119IdentityIdentity_118:output:0^NoOp_1*
T0*
_output_shapes
: Ю
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
_output_shapes
 "%
identity_119Identity_119:output:0*(
_construction_contextkEagerRuntime*
_input_shapesё
ю: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_992(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%v!

_user_specified_namecount:%u!

_user_specified_nametotal:5t1
/
_user_specified_nameAdam/v/conv3d_18/bias:5s1
/
_user_specified_nameAdam/m/conv3d_18/bias:7r3
1
_user_specified_nameAdam/v/conv3d_18/kernel:7q3
1
_user_specified_nameAdam/m/conv3d_18/kernel:5p1
/
_user_specified_nameAdam/v/conv3d_17/bias:5o1
/
_user_specified_nameAdam/m/conv3d_17/bias:7n3
1
_user_specified_nameAdam/v/conv3d_17/kernel:7m3
1
_user_specified_nameAdam/m/conv3d_17/kernel:5l1
/
_user_specified_nameAdam/v/conv3d_16/bias:5k1
/
_user_specified_nameAdam/m/conv3d_16/bias:7j3
1
_user_specified_nameAdam/v/conv3d_16/kernel:7i3
1
_user_specified_nameAdam/m/conv3d_16/kernel:5h1
/
_user_specified_nameAdam/v/conv3d_15/bias:5g1
/
_user_specified_nameAdam/m/conv3d_15/bias:7f3
1
_user_specified_nameAdam/v/conv3d_15/kernel:7e3
1
_user_specified_nameAdam/m/conv3d_15/kernel:4d0
.
_user_specified_nameAdam/v/conv3d_1/bias:4c0
.
_user_specified_nameAdam/m/conv3d_1/bias:6b2
0
_user_specified_nameAdam/v/conv3d_1/kernel:6a2
0
_user_specified_nameAdam/m/conv3d_1/kernel:2`.
,
_user_specified_nameAdam/v/conv3d/bias:2_.
,
_user_specified_nameAdam/m/conv3d/bias:4^0
.
_user_specified_nameAdam/v/conv3d/kernel:4]0
.
_user_specified_nameAdam/m/conv3d/kernel:5\1
/
_user_specified_nameAdam/v/conv3d_14/bias:5[1
/
_user_specified_nameAdam/m/conv3d_14/bias:7Z3
1
_user_specified_nameAdam/v/conv3d_14/kernel:7Y3
1
_user_specified_nameAdam/m/conv3d_14/kernel:5X1
/
_user_specified_nameAdam/v/conv3d_13/bias:5W1
/
_user_specified_nameAdam/m/conv3d_13/bias:7V3
1
_user_specified_nameAdam/v/conv3d_13/kernel:7U3
1
_user_specified_nameAdam/m/conv3d_13/kernel:4T0
.
_user_specified_nameAdam/v/conv3d_3/bias:4S0
.
_user_specified_nameAdam/m/conv3d_3/bias:6R2
0
_user_specified_nameAdam/v/conv3d_3/kernel:6Q2
0
_user_specified_nameAdam/m/conv3d_3/kernel:4P0
.
_user_specified_nameAdam/v/conv3d_2/bias:4O0
.
_user_specified_nameAdam/m/conv3d_2/bias:6N2
0
_user_specified_nameAdam/v/conv3d_2/kernel:6M2
0
_user_specified_nameAdam/m/conv3d_2/kernel:5L1
/
_user_specified_nameAdam/v/conv3d_12/bias:5K1
/
_user_specified_nameAdam/m/conv3d_12/bias:7J3
1
_user_specified_nameAdam/v/conv3d_12/kernel:7I3
1
_user_specified_nameAdam/m/conv3d_12/kernel:5H1
/
_user_specified_nameAdam/v/conv3d_11/bias:5G1
/
_user_specified_nameAdam/m/conv3d_11/bias:7F3
1
_user_specified_nameAdam/v/conv3d_11/kernel:7E3
1
_user_specified_nameAdam/m/conv3d_11/kernel:4D0
.
_user_specified_nameAdam/v/conv3d_5/bias:4C0
.
_user_specified_nameAdam/m/conv3d_5/bias:6B2
0
_user_specified_nameAdam/v/conv3d_5/kernel:6A2
0
_user_specified_nameAdam/m/conv3d_5/kernel:4@0
.
_user_specified_nameAdam/v/conv3d_4/bias:4?0
.
_user_specified_nameAdam/m/conv3d_4/bias:6>2
0
_user_specified_nameAdam/v/conv3d_4/kernel:6=2
0
_user_specified_nameAdam/m/conv3d_4/kernel:5<1
/
_user_specified_nameAdam/v/conv3d_10/bias:5;1
/
_user_specified_nameAdam/m/conv3d_10/bias:7:3
1
_user_specified_nameAdam/v/conv3d_10/kernel:793
1
_user_specified_nameAdam/m/conv3d_10/kernel:480
.
_user_specified_nameAdam/v/conv3d_9/bias:470
.
_user_specified_nameAdam/m/conv3d_9/bias:662
0
_user_specified_nameAdam/v/conv3d_9/kernel:652
0
_user_specified_nameAdam/m/conv3d_9/kernel:440
.
_user_specified_nameAdam/v/conv3d_8/bias:430
.
_user_specified_nameAdam/m/conv3d_8/bias:622
0
_user_specified_nameAdam/v/conv3d_8/kernel:612
0
_user_specified_nameAdam/m/conv3d_8/kernel:400
.
_user_specified_nameAdam/v/conv3d_7/bias:4/0
.
_user_specified_nameAdam/m/conv3d_7/bias:6.2
0
_user_specified_nameAdam/v/conv3d_7/kernel:6-2
0
_user_specified_nameAdam/m/conv3d_7/kernel:4,0
.
_user_specified_nameAdam/v/conv3d_6/bias:4+0
.
_user_specified_nameAdam/m/conv3d_6/bias:6*2
0
_user_specified_nameAdam/v/conv3d_6/kernel:6)2
0
_user_specified_nameAdam/m/conv3d_6/kernel:-()
'
_user_specified_namelearning_rate:)'%
#
_user_specified_name	iteration:.&*
(
_user_specified_nameconv3d_18/bias:0%,
*
_user_specified_nameconv3d_18/kernel:.$*
(
_user_specified_nameconv3d_17/bias:0#,
*
_user_specified_nameconv3d_17/kernel:."*
(
_user_specified_nameconv3d_16/bias:0!,
*
_user_specified_nameconv3d_16/kernel:. *
(
_user_specified_nameconv3d_15/bias:0,
*
_user_specified_nameconv3d_15/kernel:-)
'
_user_specified_nameconv3d_1/bias:/+
)
_user_specified_nameconv3d_1/kernel:+'
%
_user_specified_nameconv3d/bias:-)
'
_user_specified_nameconv3d/kernel:.*
(
_user_specified_nameconv3d_14/bias:0,
*
_user_specified_nameconv3d_14/kernel:.*
(
_user_specified_nameconv3d_13/bias:0,
*
_user_specified_nameconv3d_13/kernel:-)
'
_user_specified_nameconv3d_3/bias:/+
)
_user_specified_nameconv3d_3/kernel:-)
'
_user_specified_nameconv3d_2/bias:/+
)
_user_specified_nameconv3d_2/kernel:.*
(
_user_specified_nameconv3d_12/bias:0,
*
_user_specified_nameconv3d_12/kernel:.*
(
_user_specified_nameconv3d_11/bias:0,
*
_user_specified_nameconv3d_11/kernel:-)
'
_user_specified_nameconv3d_5/bias:/+
)
_user_specified_nameconv3d_5/kernel:-)
'
_user_specified_nameconv3d_4/bias:/+
)
_user_specified_nameconv3d_4/kernel:.
*
(
_user_specified_nameconv3d_10/bias:0	,
*
_user_specified_nameconv3d_10/kernel:-)
'
_user_specified_nameconv3d_9/bias:/+
)
_user_specified_nameconv3d_9/kernel:-)
'
_user_specified_nameconv3d_8/bias:/+
)
_user_specified_nameconv3d_8/kernel:-)
'
_user_specified_nameconv3d_7/bias:/+
)
_user_specified_nameconv3d_7/kernel:-)
'
_user_specified_nameconv3d_6/bias:/+
)
_user_specified_nameconv3d_6/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ћ
J
"__inference__update_step_xla_17339
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
й
ў
A__inference_conv3d_layer_call_and_return_conditional_losses_17901

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџаn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџаS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџа: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_17239
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
$
d
H__inference_up_sampling3d_layer_call_and_return_conditional_losses_17619

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :з
splitSplitsplit/split_dim:output:0inputs*
T0*
_output_shapesї
є:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Б
concatConcatV2split:output:0split:output:1split:output:2split:output:3split:output:4split:output:5split:output:6split:output:7split:output:8split:output:9split:output:10split:output:11concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ђ
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*Ш
_output_shapesЕ
В:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :ѓ
concat_1ConcatV2split_1:output:0split_1:output:1split_1:output:2split_1:output:3split_1:output:4split_1:output:5split_1:output:6split_1:output:7split_1:output:8split_1:output:9split_1:output:10split_1:output:11split_1:output:12split_1:output:13concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*М
_output_shapesЉ
І:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :Л
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15split_2:output:16split_2:output:16split_2:output:17split_2:output:17split_2:output:18split_2:output:18split_2:output:19split_2:output:19split_2:output:20split_2:output:20split_2:output:21split_2:output:21split_2:output:22split_2:output:22split_2:output:23split_2:output:23split_2:output:24split_2:output:24split_2:output:25split_2:output:25concat_2/axis:output:0*
N4*
T0*3
_output_shapes!
:џџџџџџџџџ4e
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
л
Z
"__inference__update_step_xla_17274
gradient&
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*+
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:T P
*
_output_shapes
:
"
_user_specified_name
gradient
а
!
 __inference__wrapped_model_15940
input_1K
-model_conv3d_6_conv3d_readvariableop_resource:<
.model_conv3d_6_biasadd_readvariableop_resource:K
-model_conv3d_7_conv3d_readvariableop_resource:<
.model_conv3d_7_biasadd_readvariableop_resource:K
-model_conv3d_8_conv3d_readvariableop_resource:<
.model_conv3d_8_biasadd_readvariableop_resource:K
-model_conv3d_9_conv3d_readvariableop_resource:<
.model_conv3d_9_biasadd_readvariableop_resource:K
-model_conv3d_4_conv3d_readvariableop_resource:<
.model_conv3d_4_biasadd_readvariableop_resource:L
.model_conv3d_10_conv3d_readvariableop_resource:=
/model_conv3d_10_biasadd_readvariableop_resource:K
-model_conv3d_5_conv3d_readvariableop_resource:<
.model_conv3d_5_biasadd_readvariableop_resource:L
.model_conv3d_11_conv3d_readvariableop_resource:=
/model_conv3d_11_biasadd_readvariableop_resource:K
-model_conv3d_2_conv3d_readvariableop_resource:<
.model_conv3d_2_biasadd_readvariableop_resource:L
.model_conv3d_12_conv3d_readvariableop_resource:=
/model_conv3d_12_biasadd_readvariableop_resource:K
-model_conv3d_3_conv3d_readvariableop_resource:<
.model_conv3d_3_biasadd_readvariableop_resource:L
.model_conv3d_13_conv3d_readvariableop_resource:=
/model_conv3d_13_biasadd_readvariableop_resource:I
+model_conv3d_conv3d_readvariableop_resource::
,model_conv3d_biasadd_readvariableop_resource:L
.model_conv3d_14_conv3d_readvariableop_resource:=
/model_conv3d_14_biasadd_readvariableop_resource:K
-model_conv3d_1_conv3d_readvariableop_resource:<
.model_conv3d_1_biasadd_readvariableop_resource:L
.model_conv3d_15_conv3d_readvariableop_resource:=
/model_conv3d_15_biasadd_readvariableop_resource:L
.model_conv3d_16_conv3d_readvariableop_resource:=
/model_conv3d_16_biasadd_readvariableop_resource:L
.model_conv3d_17_conv3d_readvariableop_resource:=
/model_conv3d_17_biasadd_readvariableop_resource:L
.model_conv3d_18_conv3d_readvariableop_resource:=
/model_conv3d_18_biasadd_readvariableop_resource:
identityЂ#model/conv3d/BiasAdd/ReadVariableOpЂ"model/conv3d/Conv3D/ReadVariableOpЂ%model/conv3d_1/BiasAdd/ReadVariableOpЂ$model/conv3d_1/Conv3D/ReadVariableOpЂ&model/conv3d_10/BiasAdd/ReadVariableOpЂ%model/conv3d_10/Conv3D/ReadVariableOpЂ&model/conv3d_11/BiasAdd/ReadVariableOpЂ%model/conv3d_11/Conv3D/ReadVariableOpЂ&model/conv3d_12/BiasAdd/ReadVariableOpЂ%model/conv3d_12/Conv3D/ReadVariableOpЂ&model/conv3d_13/BiasAdd/ReadVariableOpЂ%model/conv3d_13/Conv3D/ReadVariableOpЂ&model/conv3d_14/BiasAdd/ReadVariableOpЂ%model/conv3d_14/Conv3D/ReadVariableOpЂ&model/conv3d_15/BiasAdd/ReadVariableOpЂ%model/conv3d_15/Conv3D/ReadVariableOpЂ&model/conv3d_16/BiasAdd/ReadVariableOpЂ%model/conv3d_16/Conv3D/ReadVariableOpЂ&model/conv3d_17/BiasAdd/ReadVariableOpЂ%model/conv3d_17/Conv3D/ReadVariableOpЂ&model/conv3d_18/BiasAdd/ReadVariableOpЂ%model/conv3d_18/Conv3D/ReadVariableOpЂ%model/conv3d_2/BiasAdd/ReadVariableOpЂ$model/conv3d_2/Conv3D/ReadVariableOpЂ%model/conv3d_3/BiasAdd/ReadVariableOpЂ$model/conv3d_3/Conv3D/ReadVariableOpЂ%model/conv3d_4/BiasAdd/ReadVariableOpЂ$model/conv3d_4/Conv3D/ReadVariableOpЂ%model/conv3d_5/BiasAdd/ReadVariableOpЂ$model/conv3d_5/Conv3D/ReadVariableOpЂ%model/conv3d_6/BiasAdd/ReadVariableOpЂ$model/conv3d_6/Conv3D/ReadVariableOpЂ%model/conv3d_7/BiasAdd/ReadVariableOpЂ$model/conv3d_7/Conv3D/ReadVariableOpЂ%model/conv3d_8/BiasAdd/ReadVariableOpЂ$model/conv3d_8/Conv3D/ReadVariableOpЂ%model/conv3d_9/BiasAdd/ReadVariableOpЂ$model/conv3d_9/Conv3D/ReadVariableOp
$model/conv3d_6/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_6_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0О
model/conv3d_6/Conv3DConv3Dinput_1,model/conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа*
paddingSAME*
strides	

%model/conv3d_6/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
model/conv3d_6/BiasAddBiasAddmodel/conv3d_6/Conv3D:output:0-model/conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа{
model/conv3d_6/ReluRelumodel/conv3d_6/BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџаЯ
#model/average_pooling3d_3/AvgPool3D	AvgPool3D!model/conv3d_6/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџh*
ksize	
*
paddingVALID*
strides	

$model/conv3d_7/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_7_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0т
model/conv3d_7/Conv3DConv3D,model/average_pooling3d_3/AvgPool3D:output:0,model/conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh*
paddingSAME*
strides	

%model/conv3d_7/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
model/conv3d_7/BiasAddBiasAddmodel/conv3d_7/Conv3D:output:0-model/conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџhz
model/conv3d_7/ReluRelumodel/conv3d_7/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџhЯ
#model/average_pooling3d_4/AvgPool3D	AvgPool3D!model/conv3d_7/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ4*
ksize	
*
paddingVALID*
strides	

$model/conv3d_8/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_8_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0т
model/conv3d_8/Conv3DConv3D,model/average_pooling3d_4/AvgPool3D:output:0,model/conv3d_8/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4*
paddingSAME*
strides	

%model/conv3d_8/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
model/conv3d_8/BiasAddBiasAddmodel/conv3d_8/Conv3D:output:0-model/conv3d_8/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4z
model/conv3d_8/ReluRelumodel/conv3d_8/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ4Е
#model/average_pooling3d_1/AvgPool3D	AvgPool3Dinput_1*
T0*3
_output_shapes!
:џџџџџџџџџh*
ksize	
*
paddingVALID*
strides	
Я
#model/average_pooling3d_5/AvgPool3D	AvgPool3D!model/conv3d_8/Relu:activations:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
ksize	
*
paddingVALID*
strides	
к
#model/average_pooling3d_2/AvgPool3D	AvgPool3D,model/average_pooling3d_1/AvgPool3D:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ4*
ksize	
*
paddingVALID*
strides	

$model/conv3d_9/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_9_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0т
model/conv3d_9/Conv3DConv3D,model/average_pooling3d_5/AvgPool3D:output:0,model/conv3d_9/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

%model/conv3d_9/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
model/conv3d_9/BiasAddBiasAddmodel/conv3d_9/Conv3D:output:0-model/conv3d_9/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџz
model/conv3d_9/ReluRelumodel/conv3d_9/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ
$model/conv3d_4/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0т
model/conv3d_4/Conv3DConv3D,model/average_pooling3d_2/AvgPool3D:output:0,model/conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4*
paddingSAME*
strides	

%model/conv3d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
model/conv3d_4/BiasAddBiasAddmodel/conv3d_4/Conv3D:output:0-model/conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4z
model/conv3d_4/ReluRelumodel/conv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ4 
%model/conv3d_10/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_10_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0й
model/conv3d_10/Conv3DConv3D!model/conv3d_9/Relu:activations:0-model/conv3d_10/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	

&model/conv3d_10/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_10/BiasAddBiasAddmodel/conv3d_10/Conv3D:output:0.model/conv3d_10/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ|
model/conv3d_10/ReluRelu model/conv3d_10/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџe
#model/up_sampling3d/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
model/up_sampling3d/splitSplit,model/up_sampling3d/split/split_dim:output:0"model/conv3d_10/Relu:activations:0*
T0*
_output_shapesї
є:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splita
model/up_sampling3d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Щ
model/up_sampling3d/concatConcatV2"model/up_sampling3d/split:output:0"model/up_sampling3d/split:output:1"model/up_sampling3d/split:output:2"model/up_sampling3d/split:output:3"model/up_sampling3d/split:output:4"model/up_sampling3d/split:output:5"model/up_sampling3d/split:output:6"model/up_sampling3d/split:output:7"model/up_sampling3d/split:output:8"model/up_sampling3d/split:output:9#model/up_sampling3d/split:output:10#model/up_sampling3d/split:output:11(model/up_sampling3d/concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџg
%model/up_sampling3d/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :о
model/up_sampling3d/split_1Split.model/up_sampling3d/split_1/split_dim:output:0#model/up_sampling3d/concat:output:0*
T0*Ш
_output_shapesЕ
В:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitc
!model/up_sampling3d/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :Г
model/up_sampling3d/concat_1ConcatV2$model/up_sampling3d/split_1:output:0$model/up_sampling3d/split_1:output:1$model/up_sampling3d/split_1:output:2$model/up_sampling3d/split_1:output:3$model/up_sampling3d/split_1:output:4$model/up_sampling3d/split_1:output:5$model/up_sampling3d/split_1:output:6$model/up_sampling3d/split_1:output:7$model/up_sampling3d/split_1:output:8$model/up_sampling3d/split_1:output:9%model/up_sampling3d/split_1:output:10%model/up_sampling3d/split_1:output:11%model/up_sampling3d/split_1:output:12%model/up_sampling3d/split_1:output:13*model/up_sampling3d/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџg
%model/up_sampling3d/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
model/up_sampling3d/split_2Split.model/up_sampling3d/split_2/split_dim:output:0%model/up_sampling3d/concat_1:output:0*
T0*М
_output_shapesЉ
І:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitc
!model/up_sampling3d/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :ѓ
model/up_sampling3d/concat_2ConcatV2$model/up_sampling3d/split_2:output:0$model/up_sampling3d/split_2:output:0$model/up_sampling3d/split_2:output:1$model/up_sampling3d/split_2:output:1$model/up_sampling3d/split_2:output:2$model/up_sampling3d/split_2:output:2$model/up_sampling3d/split_2:output:3$model/up_sampling3d/split_2:output:3$model/up_sampling3d/split_2:output:4$model/up_sampling3d/split_2:output:4$model/up_sampling3d/split_2:output:5$model/up_sampling3d/split_2:output:5$model/up_sampling3d/split_2:output:6$model/up_sampling3d/split_2:output:6$model/up_sampling3d/split_2:output:7$model/up_sampling3d/split_2:output:7$model/up_sampling3d/split_2:output:8$model/up_sampling3d/split_2:output:8$model/up_sampling3d/split_2:output:9$model/up_sampling3d/split_2:output:9%model/up_sampling3d/split_2:output:10%model/up_sampling3d/split_2:output:10%model/up_sampling3d/split_2:output:11%model/up_sampling3d/split_2:output:11%model/up_sampling3d/split_2:output:12%model/up_sampling3d/split_2:output:12%model/up_sampling3d/split_2:output:13%model/up_sampling3d/split_2:output:13%model/up_sampling3d/split_2:output:14%model/up_sampling3d/split_2:output:14%model/up_sampling3d/split_2:output:15%model/up_sampling3d/split_2:output:15%model/up_sampling3d/split_2:output:16%model/up_sampling3d/split_2:output:16%model/up_sampling3d/split_2:output:17%model/up_sampling3d/split_2:output:17%model/up_sampling3d/split_2:output:18%model/up_sampling3d/split_2:output:18%model/up_sampling3d/split_2:output:19%model/up_sampling3d/split_2:output:19%model/up_sampling3d/split_2:output:20%model/up_sampling3d/split_2:output:20%model/up_sampling3d/split_2:output:21%model/up_sampling3d/split_2:output:21%model/up_sampling3d/split_2:output:22%model/up_sampling3d/split_2:output:22%model/up_sampling3d/split_2:output:23%model/up_sampling3d/split_2:output:23%model/up_sampling3d/split_2:output:24%model/up_sampling3d/split_2:output:24%model/up_sampling3d/split_2:output:25%model/up_sampling3d/split_2:output:25*model/up_sampling3d/concat_2/axis:output:0*
N4*
T0*3
_output_shapes!
:џџџџџџџџџ4
$model/conv3d_5/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0з
model/conv3d_5/Conv3DConv3D!model/conv3d_4/Relu:activations:0,model/conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4*
paddingSAME*
strides	

%model/conv3d_5/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
model/conv3d_5/BiasAddBiasAddmodel/conv3d_5/Conv3D:output:0-model/conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4z
model/conv3d_5/ReluRelumodel/conv3d_5/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ4_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :н
model/concatenate/concatConcatV2%model/up_sampling3d/concat_2:output:0!model/conv3d_5/Relu:activations:0&model/concatenate/concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџ4 
%model/conv3d_11/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_11_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0й
model/conv3d_11/Conv3DConv3D!model/concatenate/concat:output:0-model/conv3d_11/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4*
paddingSAME*
strides	

&model/conv3d_11/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_11/BiasAddBiasAddmodel/conv3d_11/Conv3D:output:0.model/conv3d_11/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4|
model/conv3d_11/ReluRelu model/conv3d_11/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ4
$model/conv3d_2/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0т
model/conv3d_2/Conv3DConv3D,model/average_pooling3d_1/AvgPool3D:output:0,model/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh*
paddingSAME*
strides	

%model/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
model/conv3d_2/BiasAddBiasAddmodel/conv3d_2/Conv3D:output:0-model/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџhz
model/conv3d_2/ReluRelumodel/conv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџh 
%model/conv3d_12/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_12_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0к
model/conv3d_12/Conv3DConv3D"model/conv3d_11/Relu:activations:0-model/conv3d_12/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4*
paddingSAME*
strides	

&model/conv3d_12/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_12/BiasAddBiasAddmodel/conv3d_12/Conv3D:output:0.model/conv3d_12/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4|
model/conv3d_12/ReluRelu model/conv3d_12/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ4g
%model/up_sampling3d_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
model/up_sampling3d_1/splitSplit.model/up_sampling3d_1/split/split_dim:output:0"model/conv3d_12/Relu:activations:0*
T0*
_output_shapesї
є:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4*
	num_splitc
!model/up_sampling3d_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :х
model/up_sampling3d_1/concatConcatV2$model/up_sampling3d_1/split:output:0$model/up_sampling3d_1/split:output:1$model/up_sampling3d_1/split:output:2$model/up_sampling3d_1/split:output:3$model/up_sampling3d_1/split:output:4$model/up_sampling3d_1/split:output:5$model/up_sampling3d_1/split:output:6$model/up_sampling3d_1/split:output:7$model/up_sampling3d_1/split:output:8$model/up_sampling3d_1/split:output:9%model/up_sampling3d_1/split:output:10%model/up_sampling3d_1/split:output:11*model/up_sampling3d_1/concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџ4i
'model/up_sampling3d_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ф
model/up_sampling3d_1/split_1Split0model/up_sampling3d_1/split_1/split_dim:output:0%model/up_sampling3d_1/concat:output:0*
T0*Ш
_output_shapesЕ
В:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4*
	num_splite
#model/up_sampling3d_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :г
model/up_sampling3d_1/concat_1ConcatV2&model/up_sampling3d_1/split_1:output:0&model/up_sampling3d_1/split_1:output:1&model/up_sampling3d_1/split_1:output:2&model/up_sampling3d_1/split_1:output:3&model/up_sampling3d_1/split_1:output:4&model/up_sampling3d_1/split_1:output:5&model/up_sampling3d_1/split_1:output:6&model/up_sampling3d_1/split_1:output:7&model/up_sampling3d_1/split_1:output:8&model/up_sampling3d_1/split_1:output:9'model/up_sampling3d_1/split_1:output:10'model/up_sampling3d_1/split_1:output:11'model/up_sampling3d_1/split_1:output:12'model/up_sampling3d_1/split_1:output:13,model/up_sampling3d_1/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџ4i
'model/up_sampling3d_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
model/up_sampling3d_1/split_2Split0model/up_sampling3d_1/split_2/split_dim:output:0'model/up_sampling3d_1/concat_1:output:0*
T0*т
_output_shapesЯ
Ь:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split4e
#model/up_sampling3d_1/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :Г"
model/up_sampling3d_1/concat_2ConcatV2&model/up_sampling3d_1/split_2:output:0&model/up_sampling3d_1/split_2:output:0&model/up_sampling3d_1/split_2:output:1&model/up_sampling3d_1/split_2:output:1&model/up_sampling3d_1/split_2:output:2&model/up_sampling3d_1/split_2:output:2&model/up_sampling3d_1/split_2:output:3&model/up_sampling3d_1/split_2:output:3&model/up_sampling3d_1/split_2:output:4&model/up_sampling3d_1/split_2:output:4&model/up_sampling3d_1/split_2:output:5&model/up_sampling3d_1/split_2:output:5&model/up_sampling3d_1/split_2:output:6&model/up_sampling3d_1/split_2:output:6&model/up_sampling3d_1/split_2:output:7&model/up_sampling3d_1/split_2:output:7&model/up_sampling3d_1/split_2:output:8&model/up_sampling3d_1/split_2:output:8&model/up_sampling3d_1/split_2:output:9&model/up_sampling3d_1/split_2:output:9'model/up_sampling3d_1/split_2:output:10'model/up_sampling3d_1/split_2:output:10'model/up_sampling3d_1/split_2:output:11'model/up_sampling3d_1/split_2:output:11'model/up_sampling3d_1/split_2:output:12'model/up_sampling3d_1/split_2:output:12'model/up_sampling3d_1/split_2:output:13'model/up_sampling3d_1/split_2:output:13'model/up_sampling3d_1/split_2:output:14'model/up_sampling3d_1/split_2:output:14'model/up_sampling3d_1/split_2:output:15'model/up_sampling3d_1/split_2:output:15'model/up_sampling3d_1/split_2:output:16'model/up_sampling3d_1/split_2:output:16'model/up_sampling3d_1/split_2:output:17'model/up_sampling3d_1/split_2:output:17'model/up_sampling3d_1/split_2:output:18'model/up_sampling3d_1/split_2:output:18'model/up_sampling3d_1/split_2:output:19'model/up_sampling3d_1/split_2:output:19'model/up_sampling3d_1/split_2:output:20'model/up_sampling3d_1/split_2:output:20'model/up_sampling3d_1/split_2:output:21'model/up_sampling3d_1/split_2:output:21'model/up_sampling3d_1/split_2:output:22'model/up_sampling3d_1/split_2:output:22'model/up_sampling3d_1/split_2:output:23'model/up_sampling3d_1/split_2:output:23'model/up_sampling3d_1/split_2:output:24'model/up_sampling3d_1/split_2:output:24'model/up_sampling3d_1/split_2:output:25'model/up_sampling3d_1/split_2:output:25'model/up_sampling3d_1/split_2:output:26'model/up_sampling3d_1/split_2:output:26'model/up_sampling3d_1/split_2:output:27'model/up_sampling3d_1/split_2:output:27'model/up_sampling3d_1/split_2:output:28'model/up_sampling3d_1/split_2:output:28'model/up_sampling3d_1/split_2:output:29'model/up_sampling3d_1/split_2:output:29'model/up_sampling3d_1/split_2:output:30'model/up_sampling3d_1/split_2:output:30'model/up_sampling3d_1/split_2:output:31'model/up_sampling3d_1/split_2:output:31'model/up_sampling3d_1/split_2:output:32'model/up_sampling3d_1/split_2:output:32'model/up_sampling3d_1/split_2:output:33'model/up_sampling3d_1/split_2:output:33'model/up_sampling3d_1/split_2:output:34'model/up_sampling3d_1/split_2:output:34'model/up_sampling3d_1/split_2:output:35'model/up_sampling3d_1/split_2:output:35'model/up_sampling3d_1/split_2:output:36'model/up_sampling3d_1/split_2:output:36'model/up_sampling3d_1/split_2:output:37'model/up_sampling3d_1/split_2:output:37'model/up_sampling3d_1/split_2:output:38'model/up_sampling3d_1/split_2:output:38'model/up_sampling3d_1/split_2:output:39'model/up_sampling3d_1/split_2:output:39'model/up_sampling3d_1/split_2:output:40'model/up_sampling3d_1/split_2:output:40'model/up_sampling3d_1/split_2:output:41'model/up_sampling3d_1/split_2:output:41'model/up_sampling3d_1/split_2:output:42'model/up_sampling3d_1/split_2:output:42'model/up_sampling3d_1/split_2:output:43'model/up_sampling3d_1/split_2:output:43'model/up_sampling3d_1/split_2:output:44'model/up_sampling3d_1/split_2:output:44'model/up_sampling3d_1/split_2:output:45'model/up_sampling3d_1/split_2:output:45'model/up_sampling3d_1/split_2:output:46'model/up_sampling3d_1/split_2:output:46'model/up_sampling3d_1/split_2:output:47'model/up_sampling3d_1/split_2:output:47'model/up_sampling3d_1/split_2:output:48'model/up_sampling3d_1/split_2:output:48'model/up_sampling3d_1/split_2:output:49'model/up_sampling3d_1/split_2:output:49'model/up_sampling3d_1/split_2:output:50'model/up_sampling3d_1/split_2:output:50'model/up_sampling3d_1/split_2:output:51'model/up_sampling3d_1/split_2:output:51,model/up_sampling3d_1/concat_2/axis:output:0*
Nh*
T0*3
_output_shapes!
:џџџџџџџџџh
$model/conv3d_3/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0з
model/conv3d_3/Conv3DConv3D!model/conv3d_2/Relu:activations:0,model/conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh*
paddingSAME*
strides	

%model/conv3d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
model/conv3d_3/BiasAddBiasAddmodel/conv3d_3/Conv3D:output:0-model/conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџhz
model/conv3d_3/ReluRelumodel/conv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџha
model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :у
model/concatenate_1/concatConcatV2'model/up_sampling3d_1/concat_2:output:0!model/conv3d_3/Relu:activations:0(model/concatenate_1/concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџh 
%model/conv3d_13/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_13_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0л
model/conv3d_13/Conv3DConv3D#model/concatenate_1/concat:output:0-model/conv3d_13/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh*
paddingSAME*
strides	

&model/conv3d_13/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_13/BiasAddBiasAddmodel/conv3d_13/Conv3D:output:0.model/conv3d_13/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh|
model/conv3d_13/ReluRelu model/conv3d_13/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџh
"model/conv3d/Conv3D/ReadVariableOpReadVariableOp+model_conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0К
model/conv3d/Conv3DConv3Dinput_1*model/conv3d/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа*
paddingSAME*
strides	

#model/conv3d/BiasAdd/ReadVariableOpReadVariableOp,model_conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
model/conv3d/BiasAddBiasAddmodel/conv3d/Conv3D:output:0+model/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџаw
model/conv3d/ReluRelumodel/conv3d/BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџа 
%model/conv3d_14/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_14_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0к
model/conv3d_14/Conv3DConv3D"model/conv3d_13/Relu:activations:0-model/conv3d_14/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh*
paddingSAME*
strides	

&model/conv3d_14/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
model/conv3d_14/BiasAddBiasAddmodel/conv3d_14/Conv3D:output:0.model/conv3d_14/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh|
model/conv3d_14/ReluRelu model/conv3d_14/BiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџhg
%model/up_sampling3d_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
model/up_sampling3d_2/splitSplit.model/up_sampling3d_2/split/split_dim:output:0"model/conv3d_14/Relu:activations:0*
T0*
_output_shapesї
є:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh*
	num_splitc
!model/up_sampling3d_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :х
model/up_sampling3d_2/concatConcatV2$model/up_sampling3d_2/split:output:0$model/up_sampling3d_2/split:output:1$model/up_sampling3d_2/split:output:2$model/up_sampling3d_2/split:output:3$model/up_sampling3d_2/split:output:4$model/up_sampling3d_2/split:output:5$model/up_sampling3d_2/split:output:6$model/up_sampling3d_2/split:output:7$model/up_sampling3d_2/split:output:8$model/up_sampling3d_2/split:output:9%model/up_sampling3d_2/split:output:10%model/up_sampling3d_2/split:output:11*model/up_sampling3d_2/concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџhi
'model/up_sampling3d_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ф
model/up_sampling3d_2/split_1Split0model/up_sampling3d_2/split_1/split_dim:output:0%model/up_sampling3d_2/concat:output:0*
T0*Ш
_output_shapesЕ
В:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh*
	num_splite
#model/up_sampling3d_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :г
model/up_sampling3d_2/concat_1ConcatV2&model/up_sampling3d_2/split_1:output:0&model/up_sampling3d_2/split_1:output:1&model/up_sampling3d_2/split_1:output:2&model/up_sampling3d_2/split_1:output:3&model/up_sampling3d_2/split_1:output:4&model/up_sampling3d_2/split_1:output:5&model/up_sampling3d_2/split_1:output:6&model/up_sampling3d_2/split_1:output:7&model/up_sampling3d_2/split_1:output:8&model/up_sampling3d_2/split_1:output:9'model/up_sampling3d_2/split_1:output:10'model/up_sampling3d_2/split_1:output:11'model/up_sampling3d_2/split_1:output:12'model/up_sampling3d_2/split_1:output:13,model/up_sampling3d_2/concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџhi
'model/up_sampling3d_2/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ь
model/up_sampling3d_2/split_2Split0model/up_sampling3d_2/split_2/split_dim:output:0'model/up_sampling3d_2/concat_1:output:0*
T0*Ў
_output_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splithe
#model/up_sampling3d_2/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :хC
model/up_sampling3d_2/concat_2ConcatV2&model/up_sampling3d_2/split_2:output:0&model/up_sampling3d_2/split_2:output:0&model/up_sampling3d_2/split_2:output:1&model/up_sampling3d_2/split_2:output:1&model/up_sampling3d_2/split_2:output:2&model/up_sampling3d_2/split_2:output:2&model/up_sampling3d_2/split_2:output:3&model/up_sampling3d_2/split_2:output:3&model/up_sampling3d_2/split_2:output:4&model/up_sampling3d_2/split_2:output:4&model/up_sampling3d_2/split_2:output:5&model/up_sampling3d_2/split_2:output:5&model/up_sampling3d_2/split_2:output:6&model/up_sampling3d_2/split_2:output:6&model/up_sampling3d_2/split_2:output:7&model/up_sampling3d_2/split_2:output:7&model/up_sampling3d_2/split_2:output:8&model/up_sampling3d_2/split_2:output:8&model/up_sampling3d_2/split_2:output:9&model/up_sampling3d_2/split_2:output:9'model/up_sampling3d_2/split_2:output:10'model/up_sampling3d_2/split_2:output:10'model/up_sampling3d_2/split_2:output:11'model/up_sampling3d_2/split_2:output:11'model/up_sampling3d_2/split_2:output:12'model/up_sampling3d_2/split_2:output:12'model/up_sampling3d_2/split_2:output:13'model/up_sampling3d_2/split_2:output:13'model/up_sampling3d_2/split_2:output:14'model/up_sampling3d_2/split_2:output:14'model/up_sampling3d_2/split_2:output:15'model/up_sampling3d_2/split_2:output:15'model/up_sampling3d_2/split_2:output:16'model/up_sampling3d_2/split_2:output:16'model/up_sampling3d_2/split_2:output:17'model/up_sampling3d_2/split_2:output:17'model/up_sampling3d_2/split_2:output:18'model/up_sampling3d_2/split_2:output:18'model/up_sampling3d_2/split_2:output:19'model/up_sampling3d_2/split_2:output:19'model/up_sampling3d_2/split_2:output:20'model/up_sampling3d_2/split_2:output:20'model/up_sampling3d_2/split_2:output:21'model/up_sampling3d_2/split_2:output:21'model/up_sampling3d_2/split_2:output:22'model/up_sampling3d_2/split_2:output:22'model/up_sampling3d_2/split_2:output:23'model/up_sampling3d_2/split_2:output:23'model/up_sampling3d_2/split_2:output:24'model/up_sampling3d_2/split_2:output:24'model/up_sampling3d_2/split_2:output:25'model/up_sampling3d_2/split_2:output:25'model/up_sampling3d_2/split_2:output:26'model/up_sampling3d_2/split_2:output:26'model/up_sampling3d_2/split_2:output:27'model/up_sampling3d_2/split_2:output:27'model/up_sampling3d_2/split_2:output:28'model/up_sampling3d_2/split_2:output:28'model/up_sampling3d_2/split_2:output:29'model/up_sampling3d_2/split_2:output:29'model/up_sampling3d_2/split_2:output:30'model/up_sampling3d_2/split_2:output:30'model/up_sampling3d_2/split_2:output:31'model/up_sampling3d_2/split_2:output:31'model/up_sampling3d_2/split_2:output:32'model/up_sampling3d_2/split_2:output:32'model/up_sampling3d_2/split_2:output:33'model/up_sampling3d_2/split_2:output:33'model/up_sampling3d_2/split_2:output:34'model/up_sampling3d_2/split_2:output:34'model/up_sampling3d_2/split_2:output:35'model/up_sampling3d_2/split_2:output:35'model/up_sampling3d_2/split_2:output:36'model/up_sampling3d_2/split_2:output:36'model/up_sampling3d_2/split_2:output:37'model/up_sampling3d_2/split_2:output:37'model/up_sampling3d_2/split_2:output:38'model/up_sampling3d_2/split_2:output:38'model/up_sampling3d_2/split_2:output:39'model/up_sampling3d_2/split_2:output:39'model/up_sampling3d_2/split_2:output:40'model/up_sampling3d_2/split_2:output:40'model/up_sampling3d_2/split_2:output:41'model/up_sampling3d_2/split_2:output:41'model/up_sampling3d_2/split_2:output:42'model/up_sampling3d_2/split_2:output:42'model/up_sampling3d_2/split_2:output:43'model/up_sampling3d_2/split_2:output:43'model/up_sampling3d_2/split_2:output:44'model/up_sampling3d_2/split_2:output:44'model/up_sampling3d_2/split_2:output:45'model/up_sampling3d_2/split_2:output:45'model/up_sampling3d_2/split_2:output:46'model/up_sampling3d_2/split_2:output:46'model/up_sampling3d_2/split_2:output:47'model/up_sampling3d_2/split_2:output:47'model/up_sampling3d_2/split_2:output:48'model/up_sampling3d_2/split_2:output:48'model/up_sampling3d_2/split_2:output:49'model/up_sampling3d_2/split_2:output:49'model/up_sampling3d_2/split_2:output:50'model/up_sampling3d_2/split_2:output:50'model/up_sampling3d_2/split_2:output:51'model/up_sampling3d_2/split_2:output:51'model/up_sampling3d_2/split_2:output:52'model/up_sampling3d_2/split_2:output:52'model/up_sampling3d_2/split_2:output:53'model/up_sampling3d_2/split_2:output:53'model/up_sampling3d_2/split_2:output:54'model/up_sampling3d_2/split_2:output:54'model/up_sampling3d_2/split_2:output:55'model/up_sampling3d_2/split_2:output:55'model/up_sampling3d_2/split_2:output:56'model/up_sampling3d_2/split_2:output:56'model/up_sampling3d_2/split_2:output:57'model/up_sampling3d_2/split_2:output:57'model/up_sampling3d_2/split_2:output:58'model/up_sampling3d_2/split_2:output:58'model/up_sampling3d_2/split_2:output:59'model/up_sampling3d_2/split_2:output:59'model/up_sampling3d_2/split_2:output:60'model/up_sampling3d_2/split_2:output:60'model/up_sampling3d_2/split_2:output:61'model/up_sampling3d_2/split_2:output:61'model/up_sampling3d_2/split_2:output:62'model/up_sampling3d_2/split_2:output:62'model/up_sampling3d_2/split_2:output:63'model/up_sampling3d_2/split_2:output:63'model/up_sampling3d_2/split_2:output:64'model/up_sampling3d_2/split_2:output:64'model/up_sampling3d_2/split_2:output:65'model/up_sampling3d_2/split_2:output:65'model/up_sampling3d_2/split_2:output:66'model/up_sampling3d_2/split_2:output:66'model/up_sampling3d_2/split_2:output:67'model/up_sampling3d_2/split_2:output:67'model/up_sampling3d_2/split_2:output:68'model/up_sampling3d_2/split_2:output:68'model/up_sampling3d_2/split_2:output:69'model/up_sampling3d_2/split_2:output:69'model/up_sampling3d_2/split_2:output:70'model/up_sampling3d_2/split_2:output:70'model/up_sampling3d_2/split_2:output:71'model/up_sampling3d_2/split_2:output:71'model/up_sampling3d_2/split_2:output:72'model/up_sampling3d_2/split_2:output:72'model/up_sampling3d_2/split_2:output:73'model/up_sampling3d_2/split_2:output:73'model/up_sampling3d_2/split_2:output:74'model/up_sampling3d_2/split_2:output:74'model/up_sampling3d_2/split_2:output:75'model/up_sampling3d_2/split_2:output:75'model/up_sampling3d_2/split_2:output:76'model/up_sampling3d_2/split_2:output:76'model/up_sampling3d_2/split_2:output:77'model/up_sampling3d_2/split_2:output:77'model/up_sampling3d_2/split_2:output:78'model/up_sampling3d_2/split_2:output:78'model/up_sampling3d_2/split_2:output:79'model/up_sampling3d_2/split_2:output:79'model/up_sampling3d_2/split_2:output:80'model/up_sampling3d_2/split_2:output:80'model/up_sampling3d_2/split_2:output:81'model/up_sampling3d_2/split_2:output:81'model/up_sampling3d_2/split_2:output:82'model/up_sampling3d_2/split_2:output:82'model/up_sampling3d_2/split_2:output:83'model/up_sampling3d_2/split_2:output:83'model/up_sampling3d_2/split_2:output:84'model/up_sampling3d_2/split_2:output:84'model/up_sampling3d_2/split_2:output:85'model/up_sampling3d_2/split_2:output:85'model/up_sampling3d_2/split_2:output:86'model/up_sampling3d_2/split_2:output:86'model/up_sampling3d_2/split_2:output:87'model/up_sampling3d_2/split_2:output:87'model/up_sampling3d_2/split_2:output:88'model/up_sampling3d_2/split_2:output:88'model/up_sampling3d_2/split_2:output:89'model/up_sampling3d_2/split_2:output:89'model/up_sampling3d_2/split_2:output:90'model/up_sampling3d_2/split_2:output:90'model/up_sampling3d_2/split_2:output:91'model/up_sampling3d_2/split_2:output:91'model/up_sampling3d_2/split_2:output:92'model/up_sampling3d_2/split_2:output:92'model/up_sampling3d_2/split_2:output:93'model/up_sampling3d_2/split_2:output:93'model/up_sampling3d_2/split_2:output:94'model/up_sampling3d_2/split_2:output:94'model/up_sampling3d_2/split_2:output:95'model/up_sampling3d_2/split_2:output:95'model/up_sampling3d_2/split_2:output:96'model/up_sampling3d_2/split_2:output:96'model/up_sampling3d_2/split_2:output:97'model/up_sampling3d_2/split_2:output:97'model/up_sampling3d_2/split_2:output:98'model/up_sampling3d_2/split_2:output:98'model/up_sampling3d_2/split_2:output:99'model/up_sampling3d_2/split_2:output:99(model/up_sampling3d_2/split_2:output:100(model/up_sampling3d_2/split_2:output:100(model/up_sampling3d_2/split_2:output:101(model/up_sampling3d_2/split_2:output:101(model/up_sampling3d_2/split_2:output:102(model/up_sampling3d_2/split_2:output:102(model/up_sampling3d_2/split_2:output:103(model/up_sampling3d_2/split_2:output:103,model/up_sampling3d_2/concat_2/axis:output:0*
Nа*
T0*4
_output_shapes"
 :џџџџџџџџџа
$model/conv3d_1/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0ж
model/conv3d_1/Conv3DConv3Dmodel/conv3d/Relu:activations:0,model/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа*
paddingSAME*
strides	

%model/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
model/conv3d_1/BiasAddBiasAddmodel/conv3d_1/Conv3D:output:0-model/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа{
model/conv3d_1/ReluRelumodel/conv3d_1/BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџаa
model/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ф
model/concatenate_2/concatConcatV2'model/up_sampling3d_2/concat_2:output:0!model/conv3d_1/Relu:activations:0(model/concatenate_2/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :џџџџџџџџџа 
%model/conv3d_15/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_15_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0м
model/conv3d_15/Conv3DConv3D#model/concatenate_2/concat:output:0-model/conv3d_15/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа*
paddingSAME*
strides	

&model/conv3d_15/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0В
model/conv3d_15/BiasAddBiasAddmodel/conv3d_15/Conv3D:output:0.model/conv3d_15/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа}
model/conv3d_15/ReluRelu model/conv3d_15/BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџа 
%model/conv3d_16/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_16_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0л
model/conv3d_16/Conv3DConv3D"model/conv3d_15/Relu:activations:0-model/conv3d_16/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа*
paddingSAME*
strides	

&model/conv3d_16/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0В
model/conv3d_16/BiasAddBiasAddmodel/conv3d_16/Conv3D:output:0.model/conv3d_16/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа}
model/conv3d_16/ReluRelu model/conv3d_16/BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџа 
%model/conv3d_17/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_17_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0л
model/conv3d_17/Conv3DConv3D"model/conv3d_16/Relu:activations:0-model/conv3d_17/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа*
paddingSAME*
strides	

&model/conv3d_17/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0В
model/conv3d_17/BiasAddBiasAddmodel/conv3d_17/Conv3D:output:0.model/conv3d_17/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа}
model/conv3d_17/ReluRelu model/conv3d_17/BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџа 
%model/conv3d_18/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_18_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0л
model/conv3d_18/Conv3DConv3D"model/conv3d_17/Relu:activations:0-model/conv3d_18/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа*
paddingSAME*
strides	

&model/conv3d_18/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0В
model/conv3d_18/BiasAddBiasAddmodel/conv3d_18/Conv3D:output:0.model/conv3d_18/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа
model/conv3d_18/SigmoidSigmoid model/conv3d_18/BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџаw
IdentityIdentitymodel/conv3d_18/Sigmoid:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџа
NoOpNoOp$^model/conv3d/BiasAdd/ReadVariableOp#^model/conv3d/Conv3D/ReadVariableOp&^model/conv3d_1/BiasAdd/ReadVariableOp%^model/conv3d_1/Conv3D/ReadVariableOp'^model/conv3d_10/BiasAdd/ReadVariableOp&^model/conv3d_10/Conv3D/ReadVariableOp'^model/conv3d_11/BiasAdd/ReadVariableOp&^model/conv3d_11/Conv3D/ReadVariableOp'^model/conv3d_12/BiasAdd/ReadVariableOp&^model/conv3d_12/Conv3D/ReadVariableOp'^model/conv3d_13/BiasAdd/ReadVariableOp&^model/conv3d_13/Conv3D/ReadVariableOp'^model/conv3d_14/BiasAdd/ReadVariableOp&^model/conv3d_14/Conv3D/ReadVariableOp'^model/conv3d_15/BiasAdd/ReadVariableOp&^model/conv3d_15/Conv3D/ReadVariableOp'^model/conv3d_16/BiasAdd/ReadVariableOp&^model/conv3d_16/Conv3D/ReadVariableOp'^model/conv3d_17/BiasAdd/ReadVariableOp&^model/conv3d_17/Conv3D/ReadVariableOp'^model/conv3d_18/BiasAdd/ReadVariableOp&^model/conv3d_18/Conv3D/ReadVariableOp&^model/conv3d_2/BiasAdd/ReadVariableOp%^model/conv3d_2/Conv3D/ReadVariableOp&^model/conv3d_3/BiasAdd/ReadVariableOp%^model/conv3d_3/Conv3D/ReadVariableOp&^model/conv3d_4/BiasAdd/ReadVariableOp%^model/conv3d_4/Conv3D/ReadVariableOp&^model/conv3d_5/BiasAdd/ReadVariableOp%^model/conv3d_5/Conv3D/ReadVariableOp&^model/conv3d_6/BiasAdd/ReadVariableOp%^model/conv3d_6/Conv3D/ReadVariableOp&^model/conv3d_7/BiasAdd/ReadVariableOp%^model/conv3d_7/Conv3D/ReadVariableOp&^model/conv3d_8/BiasAdd/ReadVariableOp%^model/conv3d_8/Conv3D/ReadVariableOp&^model/conv3d_9/BiasAdd/ReadVariableOp%^model/conv3d_9/Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:џџџџџџџџџа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#model/conv3d/BiasAdd/ReadVariableOp#model/conv3d/BiasAdd/ReadVariableOp2H
"model/conv3d/Conv3D/ReadVariableOp"model/conv3d/Conv3D/ReadVariableOp2N
%model/conv3d_1/BiasAdd/ReadVariableOp%model/conv3d_1/BiasAdd/ReadVariableOp2L
$model/conv3d_1/Conv3D/ReadVariableOp$model/conv3d_1/Conv3D/ReadVariableOp2P
&model/conv3d_10/BiasAdd/ReadVariableOp&model/conv3d_10/BiasAdd/ReadVariableOp2N
%model/conv3d_10/Conv3D/ReadVariableOp%model/conv3d_10/Conv3D/ReadVariableOp2P
&model/conv3d_11/BiasAdd/ReadVariableOp&model/conv3d_11/BiasAdd/ReadVariableOp2N
%model/conv3d_11/Conv3D/ReadVariableOp%model/conv3d_11/Conv3D/ReadVariableOp2P
&model/conv3d_12/BiasAdd/ReadVariableOp&model/conv3d_12/BiasAdd/ReadVariableOp2N
%model/conv3d_12/Conv3D/ReadVariableOp%model/conv3d_12/Conv3D/ReadVariableOp2P
&model/conv3d_13/BiasAdd/ReadVariableOp&model/conv3d_13/BiasAdd/ReadVariableOp2N
%model/conv3d_13/Conv3D/ReadVariableOp%model/conv3d_13/Conv3D/ReadVariableOp2P
&model/conv3d_14/BiasAdd/ReadVariableOp&model/conv3d_14/BiasAdd/ReadVariableOp2N
%model/conv3d_14/Conv3D/ReadVariableOp%model/conv3d_14/Conv3D/ReadVariableOp2P
&model/conv3d_15/BiasAdd/ReadVariableOp&model/conv3d_15/BiasAdd/ReadVariableOp2N
%model/conv3d_15/Conv3D/ReadVariableOp%model/conv3d_15/Conv3D/ReadVariableOp2P
&model/conv3d_16/BiasAdd/ReadVariableOp&model/conv3d_16/BiasAdd/ReadVariableOp2N
%model/conv3d_16/Conv3D/ReadVariableOp%model/conv3d_16/Conv3D/ReadVariableOp2P
&model/conv3d_17/BiasAdd/ReadVariableOp&model/conv3d_17/BiasAdd/ReadVariableOp2N
%model/conv3d_17/Conv3D/ReadVariableOp%model/conv3d_17/Conv3D/ReadVariableOp2P
&model/conv3d_18/BiasAdd/ReadVariableOp&model/conv3d_18/BiasAdd/ReadVariableOp2N
%model/conv3d_18/Conv3D/ReadVariableOp%model/conv3d_18/Conv3D/ReadVariableOp2N
%model/conv3d_2/BiasAdd/ReadVariableOp%model/conv3d_2/BiasAdd/ReadVariableOp2L
$model/conv3d_2/Conv3D/ReadVariableOp$model/conv3d_2/Conv3D/ReadVariableOp2N
%model/conv3d_3/BiasAdd/ReadVariableOp%model/conv3d_3/BiasAdd/ReadVariableOp2L
$model/conv3d_3/Conv3D/ReadVariableOp$model/conv3d_3/Conv3D/ReadVariableOp2N
%model/conv3d_4/BiasAdd/ReadVariableOp%model/conv3d_4/BiasAdd/ReadVariableOp2L
$model/conv3d_4/Conv3D/ReadVariableOp$model/conv3d_4/Conv3D/ReadVariableOp2N
%model/conv3d_5/BiasAdd/ReadVariableOp%model/conv3d_5/BiasAdd/ReadVariableOp2L
$model/conv3d_5/Conv3D/ReadVariableOp$model/conv3d_5/Conv3D/ReadVariableOp2N
%model/conv3d_6/BiasAdd/ReadVariableOp%model/conv3d_6/BiasAdd/ReadVariableOp2L
$model/conv3d_6/Conv3D/ReadVariableOp$model/conv3d_6/Conv3D/ReadVariableOp2N
%model/conv3d_7/BiasAdd/ReadVariableOp%model/conv3d_7/BiasAdd/ReadVariableOp2L
$model/conv3d_7/Conv3D/ReadVariableOp$model/conv3d_7/Conv3D/ReadVariableOp2N
%model/conv3d_8/BiasAdd/ReadVariableOp%model/conv3d_8/BiasAdd/ReadVariableOp2L
$model/conv3d_8/Conv3D/ReadVariableOp$model/conv3d_8/Conv3D/ReadVariableOp2N
%model/conv3d_9/BiasAdd/ReadVariableOp%model/conv3d_9/BiasAdd/ReadVariableOp2L
$model/conv3d_9/Conv3D/ReadVariableOp$model/conv3d_9/Conv3D/ReadVariableOp:(&$
"
_user_specified_name
resource:(%$
"
_user_specified_name
resource:($$
"
_user_specified_name
resource:(#$
"
_user_specified_name
resource:("$
"
_user_specified_name
resource:(!$
"
_user_specified_name
resource:( $
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
4
_output_shapes"
 :џџџџџџџџџа
!
_user_specified_name	input_1
м
j
N__inference_average_pooling3d_4_layer_call_and_return_conditional_losses_15955

inputs
identityО
	AvgPool3D	AvgPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingVALID*
strides	

IdentityIdentityAvgPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
л
Z
"__inference__update_step_xla_17304
gradient&
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*+
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:T P
*
_output_shapes
:
"
_user_specified_name
gradient
м
K
/__inference_up_sampling3d_1_layer_call_fn_17717

inputs
identityФ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_16322l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџh"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ4:[ W
3
_output_shapes!
:џџџџџџџџџ4
 
_user_specified_nameinputs
л
Z
"__inference__update_step_xla_17334
gradient&
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*+
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:T P
*
_output_shapes
:
"
_user_specified_name
gradient
л
Z
"__inference__update_step_xla_17194
gradient&
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*+
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:T P
*
_output_shapes
:
"
_user_specified_name
gradient
ћ
Y
-__inference_concatenate_2_layer_call_fn_18075
inputs_0
inputs_1
identityа
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_16562m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџа"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџа:џџџџџџџџџа:^Z
4
_output_shapes"
 :џџџџџџџџџа
"
_user_specified_name
inputs_1:^ Z
4
_output_shapes"
 :џџџџџџџџџа
"
_user_specified_name
inputs_0
ѕ
O
3__inference_average_pooling3d_5_layer_call_fn_17464

inputs
identityь
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling3d_5_layer_call_and_return_conditional_losses_15965
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
л
Z
"__inference__update_step_xla_17264
gradient&
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*+
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:T P
*
_output_shapes
:
"
_user_specified_name
gradient
л

C__inference_conv3d_1_layer_call_and_return_conditional_losses_18069

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџаn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџаS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџа: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs
е

C__inference_conv3d_4_layer_call_and_return_conditional_losses_17549

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ4m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ4S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџ4
 
_user_specified_nameinputs
ѕ
O
3__inference_average_pooling3d_1_layer_call_fn_17474

inputs
identityь
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling3d_1_layer_call_and_return_conditional_losses_15975
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
е

C__inference_conv3d_7_layer_call_and_return_conditional_losses_16020

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџhm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџhS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџh
 
_user_specified_nameinputs
ж

D__inference_conv3d_12_layer_call_and_return_conditional_losses_17692

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ4m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ4S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџ4
 
_user_specified_nameinputs
2
f
J__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_16322

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :з
splitSplitsplit/split_dim:output:0inputs*
T0*
_output_shapesї
є:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Б
concatConcatV2split:output:0split:output:1split:output:2split:output:3split:output:4split:output:5split:output:6split:output:7split:output:8split:output:9split:output:10split:output:11concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџ4S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ђ
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*Ш
_output_shapesЕ
В:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4*
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :ѓ
concat_1ConcatV2split_1:output:0split_1:output:1split_1:output:2split_1:output:3split_1:output:4split_1:output:5split_1:output:6split_1:output:7split_1:output:8split_1:output:9split_1:output:10split_1:output:11split_1:output:12split_1:output:13concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџ4S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :О
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*т
_output_shapesЯ
Ь:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split4O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15split_2:output:16split_2:output:16split_2:output:17split_2:output:17split_2:output:18split_2:output:18split_2:output:19split_2:output:19split_2:output:20split_2:output:20split_2:output:21split_2:output:21split_2:output:22split_2:output:22split_2:output:23split_2:output:23split_2:output:24split_2:output:24split_2:output:25split_2:output:25split_2:output:26split_2:output:26split_2:output:27split_2:output:27split_2:output:28split_2:output:28split_2:output:29split_2:output:29split_2:output:30split_2:output:30split_2:output:31split_2:output:31split_2:output:32split_2:output:32split_2:output:33split_2:output:33split_2:output:34split_2:output:34split_2:output:35split_2:output:35split_2:output:36split_2:output:36split_2:output:37split_2:output:37split_2:output:38split_2:output:38split_2:output:39split_2:output:39split_2:output:40split_2:output:40split_2:output:41split_2:output:41split_2:output:42split_2:output:42split_2:output:43split_2:output:43split_2:output:44split_2:output:44split_2:output:45split_2:output:45split_2:output:46split_2:output:46split_2:output:47split_2:output:47split_2:output:48split_2:output:48split_2:output:49split_2:output:49split_2:output:50split_2:output:50split_2:output:51split_2:output:51concat_2/axis:output:0*
Nh*
T0*3
_output_shapes!
:џџџџџџџџџhe
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:џџџџџџџџџh"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ4:[ W
3
_output_shapes!
:џџџџџџџџџ4
 
_user_specified_nameinputs
Љ
Ё
(__inference_conv3d_3_layer_call_fn_17817

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_16334{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџh<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџh: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name17813:%!

_user_specified_name17811:[ W
3
_output_shapes!
:џџџџџџџџџh
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_17349
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
ЊN
f
J__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_18049

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :з
splitSplitsplit/split_dim:output:0inputs*
T0*
_output_shapesї
є:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Б
concatConcatV2split:output:0split:output:1split:output:2split:output:3split:output:4split:output:5split:output:6split:output:7split:output:8split:output:9split:output:10split:output:11concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџhS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ђ
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*Ш
_output_shapesЕ
В:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh*
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :ѓ
concat_1ConcatV2split_1:output:0split_1:output:1split_1:output:2split_1:output:3split_1:output:4split_1:output:5split_1:output:6split_1:output:7split_1:output:8split_1:output:9split_1:output:10split_1:output:11split_1:output:12split_1:output:13concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџhS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*Ў
_output_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splithO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :й
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15split_2:output:16split_2:output:16split_2:output:17split_2:output:17split_2:output:18split_2:output:18split_2:output:19split_2:output:19split_2:output:20split_2:output:20split_2:output:21split_2:output:21split_2:output:22split_2:output:22split_2:output:23split_2:output:23split_2:output:24split_2:output:24split_2:output:25split_2:output:25split_2:output:26split_2:output:26split_2:output:27split_2:output:27split_2:output:28split_2:output:28split_2:output:29split_2:output:29split_2:output:30split_2:output:30split_2:output:31split_2:output:31split_2:output:32split_2:output:32split_2:output:33split_2:output:33split_2:output:34split_2:output:34split_2:output:35split_2:output:35split_2:output:36split_2:output:36split_2:output:37split_2:output:37split_2:output:38split_2:output:38split_2:output:39split_2:output:39split_2:output:40split_2:output:40split_2:output:41split_2:output:41split_2:output:42split_2:output:42split_2:output:43split_2:output:43split_2:output:44split_2:output:44split_2:output:45split_2:output:45split_2:output:46split_2:output:46split_2:output:47split_2:output:47split_2:output:48split_2:output:48split_2:output:49split_2:output:49split_2:output:50split_2:output:50split_2:output:51split_2:output:51split_2:output:52split_2:output:52split_2:output:53split_2:output:53split_2:output:54split_2:output:54split_2:output:55split_2:output:55split_2:output:56split_2:output:56split_2:output:57split_2:output:57split_2:output:58split_2:output:58split_2:output:59split_2:output:59split_2:output:60split_2:output:60split_2:output:61split_2:output:61split_2:output:62split_2:output:62split_2:output:63split_2:output:63split_2:output:64split_2:output:64split_2:output:65split_2:output:65split_2:output:66split_2:output:66split_2:output:67split_2:output:67split_2:output:68split_2:output:68split_2:output:69split_2:output:69split_2:output:70split_2:output:70split_2:output:71split_2:output:71split_2:output:72split_2:output:72split_2:output:73split_2:output:73split_2:output:74split_2:output:74split_2:output:75split_2:output:75split_2:output:76split_2:output:76split_2:output:77split_2:output:77split_2:output:78split_2:output:78split_2:output:79split_2:output:79split_2:output:80split_2:output:80split_2:output:81split_2:output:81split_2:output:82split_2:output:82split_2:output:83split_2:output:83split_2:output:84split_2:output:84split_2:output:85split_2:output:85split_2:output:86split_2:output:86split_2:output:87split_2:output:87split_2:output:88split_2:output:88split_2:output:89split_2:output:89split_2:output:90split_2:output:90split_2:output:91split_2:output:91split_2:output:92split_2:output:92split_2:output:93split_2:output:93split_2:output:94split_2:output:94split_2:output:95split_2:output:95split_2:output:96split_2:output:96split_2:output:97split_2:output:97split_2:output:98split_2:output:98split_2:output:99split_2:output:99split_2:output:100split_2:output:100split_2:output:101split_2:output:101split_2:output:102split_2:output:102split_2:output:103split_2:output:103concat_2/axis:output:0*
Nа*
T0*4
_output_shapes"
 :џџџџџџџџџаf
IdentityIdentityconcat_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџа"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџh:[ W
3
_output_shapes!
:џџџџџџџџџh
 
_user_specified_nameinputs
З
уp
__inference__traced_save_18892
file_prefixD
&read_disablecopyonread_conv3d_6_kernel:4
&read_1_disablecopyonread_conv3d_6_bias:F
(read_2_disablecopyonread_conv3d_7_kernel:4
&read_3_disablecopyonread_conv3d_7_bias:F
(read_4_disablecopyonread_conv3d_8_kernel:4
&read_5_disablecopyonread_conv3d_8_bias:F
(read_6_disablecopyonread_conv3d_9_kernel:4
&read_7_disablecopyonread_conv3d_9_bias:G
)read_8_disablecopyonread_conv3d_10_kernel:5
'read_9_disablecopyonread_conv3d_10_bias:G
)read_10_disablecopyonread_conv3d_4_kernel:5
'read_11_disablecopyonread_conv3d_4_bias:G
)read_12_disablecopyonread_conv3d_5_kernel:5
'read_13_disablecopyonread_conv3d_5_bias:H
*read_14_disablecopyonread_conv3d_11_kernel:6
(read_15_disablecopyonread_conv3d_11_bias:H
*read_16_disablecopyonread_conv3d_12_kernel:6
(read_17_disablecopyonread_conv3d_12_bias:G
)read_18_disablecopyonread_conv3d_2_kernel:5
'read_19_disablecopyonread_conv3d_2_bias:G
)read_20_disablecopyonread_conv3d_3_kernel:5
'read_21_disablecopyonread_conv3d_3_bias:H
*read_22_disablecopyonread_conv3d_13_kernel:6
(read_23_disablecopyonread_conv3d_13_bias:H
*read_24_disablecopyonread_conv3d_14_kernel:6
(read_25_disablecopyonread_conv3d_14_bias:E
'read_26_disablecopyonread_conv3d_kernel:3
%read_27_disablecopyonread_conv3d_bias:G
)read_28_disablecopyonread_conv3d_1_kernel:5
'read_29_disablecopyonread_conv3d_1_bias:H
*read_30_disablecopyonread_conv3d_15_kernel:6
(read_31_disablecopyonread_conv3d_15_bias:H
*read_32_disablecopyonread_conv3d_16_kernel:6
(read_33_disablecopyonread_conv3d_16_bias:H
*read_34_disablecopyonread_conv3d_17_kernel:6
(read_35_disablecopyonread_conv3d_17_bias:H
*read_36_disablecopyonread_conv3d_18_kernel:6
(read_37_disablecopyonread_conv3d_18_bias:-
#read_38_disablecopyonread_iteration:	 1
'read_39_disablecopyonread_learning_rate: N
0read_40_disablecopyonread_adam_m_conv3d_6_kernel:N
0read_41_disablecopyonread_adam_v_conv3d_6_kernel:<
.read_42_disablecopyonread_adam_m_conv3d_6_bias:<
.read_43_disablecopyonread_adam_v_conv3d_6_bias:N
0read_44_disablecopyonread_adam_m_conv3d_7_kernel:N
0read_45_disablecopyonread_adam_v_conv3d_7_kernel:<
.read_46_disablecopyonread_adam_m_conv3d_7_bias:<
.read_47_disablecopyonread_adam_v_conv3d_7_bias:N
0read_48_disablecopyonread_adam_m_conv3d_8_kernel:N
0read_49_disablecopyonread_adam_v_conv3d_8_kernel:<
.read_50_disablecopyonread_adam_m_conv3d_8_bias:<
.read_51_disablecopyonread_adam_v_conv3d_8_bias:N
0read_52_disablecopyonread_adam_m_conv3d_9_kernel:N
0read_53_disablecopyonread_adam_v_conv3d_9_kernel:<
.read_54_disablecopyonread_adam_m_conv3d_9_bias:<
.read_55_disablecopyonread_adam_v_conv3d_9_bias:O
1read_56_disablecopyonread_adam_m_conv3d_10_kernel:O
1read_57_disablecopyonread_adam_v_conv3d_10_kernel:=
/read_58_disablecopyonread_adam_m_conv3d_10_bias:=
/read_59_disablecopyonread_adam_v_conv3d_10_bias:N
0read_60_disablecopyonread_adam_m_conv3d_4_kernel:N
0read_61_disablecopyonread_adam_v_conv3d_4_kernel:<
.read_62_disablecopyonread_adam_m_conv3d_4_bias:<
.read_63_disablecopyonread_adam_v_conv3d_4_bias:N
0read_64_disablecopyonread_adam_m_conv3d_5_kernel:N
0read_65_disablecopyonread_adam_v_conv3d_5_kernel:<
.read_66_disablecopyonread_adam_m_conv3d_5_bias:<
.read_67_disablecopyonread_adam_v_conv3d_5_bias:O
1read_68_disablecopyonread_adam_m_conv3d_11_kernel:O
1read_69_disablecopyonread_adam_v_conv3d_11_kernel:=
/read_70_disablecopyonread_adam_m_conv3d_11_bias:=
/read_71_disablecopyonread_adam_v_conv3d_11_bias:O
1read_72_disablecopyonread_adam_m_conv3d_12_kernel:O
1read_73_disablecopyonread_adam_v_conv3d_12_kernel:=
/read_74_disablecopyonread_adam_m_conv3d_12_bias:=
/read_75_disablecopyonread_adam_v_conv3d_12_bias:N
0read_76_disablecopyonread_adam_m_conv3d_2_kernel:N
0read_77_disablecopyonread_adam_v_conv3d_2_kernel:<
.read_78_disablecopyonread_adam_m_conv3d_2_bias:<
.read_79_disablecopyonread_adam_v_conv3d_2_bias:N
0read_80_disablecopyonread_adam_m_conv3d_3_kernel:N
0read_81_disablecopyonread_adam_v_conv3d_3_kernel:<
.read_82_disablecopyonread_adam_m_conv3d_3_bias:<
.read_83_disablecopyonread_adam_v_conv3d_3_bias:O
1read_84_disablecopyonread_adam_m_conv3d_13_kernel:O
1read_85_disablecopyonread_adam_v_conv3d_13_kernel:=
/read_86_disablecopyonread_adam_m_conv3d_13_bias:=
/read_87_disablecopyonread_adam_v_conv3d_13_bias:O
1read_88_disablecopyonread_adam_m_conv3d_14_kernel:O
1read_89_disablecopyonread_adam_v_conv3d_14_kernel:=
/read_90_disablecopyonread_adam_m_conv3d_14_bias:=
/read_91_disablecopyonread_adam_v_conv3d_14_bias:L
.read_92_disablecopyonread_adam_m_conv3d_kernel:L
.read_93_disablecopyonread_adam_v_conv3d_kernel::
,read_94_disablecopyonread_adam_m_conv3d_bias::
,read_95_disablecopyonread_adam_v_conv3d_bias:N
0read_96_disablecopyonread_adam_m_conv3d_1_kernel:N
0read_97_disablecopyonread_adam_v_conv3d_1_kernel:<
.read_98_disablecopyonread_adam_m_conv3d_1_bias:<
.read_99_disablecopyonread_adam_v_conv3d_1_bias:P
2read_100_disablecopyonread_adam_m_conv3d_15_kernel:P
2read_101_disablecopyonread_adam_v_conv3d_15_kernel:>
0read_102_disablecopyonread_adam_m_conv3d_15_bias:>
0read_103_disablecopyonread_adam_v_conv3d_15_bias:P
2read_104_disablecopyonread_adam_m_conv3d_16_kernel:P
2read_105_disablecopyonread_adam_v_conv3d_16_kernel:>
0read_106_disablecopyonread_adam_m_conv3d_16_bias:>
0read_107_disablecopyonread_adam_v_conv3d_16_bias:P
2read_108_disablecopyonread_adam_m_conv3d_17_kernel:P
2read_109_disablecopyonread_adam_v_conv3d_17_kernel:>
0read_110_disablecopyonread_adam_m_conv3d_17_bias:>
0read_111_disablecopyonread_adam_v_conv3d_17_bias:P
2read_112_disablecopyonread_adam_m_conv3d_18_kernel:P
2read_113_disablecopyonread_adam_v_conv3d_18_kernel:>
0read_114_disablecopyonread_adam_m_conv3d_18_bias:>
0read_115_disablecopyonread_adam_v_conv3d_18_bias:*
 read_116_disablecopyonread_total: *
 read_117_disablecopyonread_count: 
savev2_const
identity_237ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_100/DisableCopyOnReadЂRead_100/ReadVariableOpЂRead_101/DisableCopyOnReadЂRead_101/ReadVariableOpЂRead_102/DisableCopyOnReadЂRead_102/ReadVariableOpЂRead_103/DisableCopyOnReadЂRead_103/ReadVariableOpЂRead_104/DisableCopyOnReadЂRead_104/ReadVariableOpЂRead_105/DisableCopyOnReadЂRead_105/ReadVariableOpЂRead_106/DisableCopyOnReadЂRead_106/ReadVariableOpЂRead_107/DisableCopyOnReadЂRead_107/ReadVariableOpЂRead_108/DisableCopyOnReadЂRead_108/ReadVariableOpЂRead_109/DisableCopyOnReadЂRead_109/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_110/DisableCopyOnReadЂRead_110/ReadVariableOpЂRead_111/DisableCopyOnReadЂRead_111/ReadVariableOpЂRead_112/DisableCopyOnReadЂRead_112/ReadVariableOpЂRead_113/DisableCopyOnReadЂRead_113/ReadVariableOpЂRead_114/DisableCopyOnReadЂRead_114/ReadVariableOpЂRead_115/DisableCopyOnReadЂRead_115/ReadVariableOpЂRead_116/DisableCopyOnReadЂRead_116/ReadVariableOpЂRead_117/DisableCopyOnReadЂRead_117/ReadVariableOpЂRead_12/DisableCopyOnReadЂRead_12/ReadVariableOpЂRead_13/DisableCopyOnReadЂRead_13/ReadVariableOpЂRead_14/DisableCopyOnReadЂRead_14/ReadVariableOpЂRead_15/DisableCopyOnReadЂRead_15/ReadVariableOpЂRead_16/DisableCopyOnReadЂRead_16/ReadVariableOpЂRead_17/DisableCopyOnReadЂRead_17/ReadVariableOpЂRead_18/DisableCopyOnReadЂRead_18/ReadVariableOpЂRead_19/DisableCopyOnReadЂRead_19/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_20/DisableCopyOnReadЂRead_20/ReadVariableOpЂRead_21/DisableCopyOnReadЂRead_21/ReadVariableOpЂRead_22/DisableCopyOnReadЂRead_22/ReadVariableOpЂRead_23/DisableCopyOnReadЂRead_23/ReadVariableOpЂRead_24/DisableCopyOnReadЂRead_24/ReadVariableOpЂRead_25/DisableCopyOnReadЂRead_25/ReadVariableOpЂRead_26/DisableCopyOnReadЂRead_26/ReadVariableOpЂRead_27/DisableCopyOnReadЂRead_27/ReadVariableOpЂRead_28/DisableCopyOnReadЂRead_28/ReadVariableOpЂRead_29/DisableCopyOnReadЂRead_29/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_30/DisableCopyOnReadЂRead_30/ReadVariableOpЂRead_31/DisableCopyOnReadЂRead_31/ReadVariableOpЂRead_32/DisableCopyOnReadЂRead_32/ReadVariableOpЂRead_33/DisableCopyOnReadЂRead_33/ReadVariableOpЂRead_34/DisableCopyOnReadЂRead_34/ReadVariableOpЂRead_35/DisableCopyOnReadЂRead_35/ReadVariableOpЂRead_36/DisableCopyOnReadЂRead_36/ReadVariableOpЂRead_37/DisableCopyOnReadЂRead_37/ReadVariableOpЂRead_38/DisableCopyOnReadЂRead_38/ReadVariableOpЂRead_39/DisableCopyOnReadЂRead_39/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_40/DisableCopyOnReadЂRead_40/ReadVariableOpЂRead_41/DisableCopyOnReadЂRead_41/ReadVariableOpЂRead_42/DisableCopyOnReadЂRead_42/ReadVariableOpЂRead_43/DisableCopyOnReadЂRead_43/ReadVariableOpЂRead_44/DisableCopyOnReadЂRead_44/ReadVariableOpЂRead_45/DisableCopyOnReadЂRead_45/ReadVariableOpЂRead_46/DisableCopyOnReadЂRead_46/ReadVariableOpЂRead_47/DisableCopyOnReadЂRead_47/ReadVariableOpЂRead_48/DisableCopyOnReadЂRead_48/ReadVariableOpЂRead_49/DisableCopyOnReadЂRead_49/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_50/DisableCopyOnReadЂRead_50/ReadVariableOpЂRead_51/DisableCopyOnReadЂRead_51/ReadVariableOpЂRead_52/DisableCopyOnReadЂRead_52/ReadVariableOpЂRead_53/DisableCopyOnReadЂRead_53/ReadVariableOpЂRead_54/DisableCopyOnReadЂRead_54/ReadVariableOpЂRead_55/DisableCopyOnReadЂRead_55/ReadVariableOpЂRead_56/DisableCopyOnReadЂRead_56/ReadVariableOpЂRead_57/DisableCopyOnReadЂRead_57/ReadVariableOpЂRead_58/DisableCopyOnReadЂRead_58/ReadVariableOpЂRead_59/DisableCopyOnReadЂRead_59/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_60/DisableCopyOnReadЂRead_60/ReadVariableOpЂRead_61/DisableCopyOnReadЂRead_61/ReadVariableOpЂRead_62/DisableCopyOnReadЂRead_62/ReadVariableOpЂRead_63/DisableCopyOnReadЂRead_63/ReadVariableOpЂRead_64/DisableCopyOnReadЂRead_64/ReadVariableOpЂRead_65/DisableCopyOnReadЂRead_65/ReadVariableOpЂRead_66/DisableCopyOnReadЂRead_66/ReadVariableOpЂRead_67/DisableCopyOnReadЂRead_67/ReadVariableOpЂRead_68/DisableCopyOnReadЂRead_68/ReadVariableOpЂRead_69/DisableCopyOnReadЂRead_69/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_70/DisableCopyOnReadЂRead_70/ReadVariableOpЂRead_71/DisableCopyOnReadЂRead_71/ReadVariableOpЂRead_72/DisableCopyOnReadЂRead_72/ReadVariableOpЂRead_73/DisableCopyOnReadЂRead_73/ReadVariableOpЂRead_74/DisableCopyOnReadЂRead_74/ReadVariableOpЂRead_75/DisableCopyOnReadЂRead_75/ReadVariableOpЂRead_76/DisableCopyOnReadЂRead_76/ReadVariableOpЂRead_77/DisableCopyOnReadЂRead_77/ReadVariableOpЂRead_78/DisableCopyOnReadЂRead_78/ReadVariableOpЂRead_79/DisableCopyOnReadЂRead_79/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_80/DisableCopyOnReadЂRead_80/ReadVariableOpЂRead_81/DisableCopyOnReadЂRead_81/ReadVariableOpЂRead_82/DisableCopyOnReadЂRead_82/ReadVariableOpЂRead_83/DisableCopyOnReadЂRead_83/ReadVariableOpЂRead_84/DisableCopyOnReadЂRead_84/ReadVariableOpЂRead_85/DisableCopyOnReadЂRead_85/ReadVariableOpЂRead_86/DisableCopyOnReadЂRead_86/ReadVariableOpЂRead_87/DisableCopyOnReadЂRead_87/ReadVariableOpЂRead_88/DisableCopyOnReadЂRead_88/ReadVariableOpЂRead_89/DisableCopyOnReadЂRead_89/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpЂRead_90/DisableCopyOnReadЂRead_90/ReadVariableOpЂRead_91/DisableCopyOnReadЂRead_91/ReadVariableOpЂRead_92/DisableCopyOnReadЂRead_92/ReadVariableOpЂRead_93/DisableCopyOnReadЂRead_93/ReadVariableOpЂRead_94/DisableCopyOnReadЂRead_94/ReadVariableOpЂRead_95/DisableCopyOnReadЂRead_95/ReadVariableOpЂRead_96/DisableCopyOnReadЂRead_96/ReadVariableOpЂRead_97/DisableCopyOnReadЂRead_97/ReadVariableOpЂRead_98/DisableCopyOnReadЂRead_98/ReadVariableOpЂRead_99/DisableCopyOnReadЂRead_99/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_conv3d_6_kernel"/device:CPU:0*
_output_shapes
 Ў
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_conv3d_6_kernel^Read/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0u
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:m

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0**
_output_shapes
:z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_conv3d_6_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_conv3d_6_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_conv3d_7_kernel"/device:CPU:0*
_output_shapes
 Д
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_conv3d_7_kernel^Read_2/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0y

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:o

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0**
_output_shapes
:z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_conv3d_7_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_conv3d_7_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_conv3d_8_kernel"/device:CPU:0*
_output_shapes
 Д
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_conv3d_8_kernel^Read_4/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0y

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:o

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0**
_output_shapes
:z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_conv3d_8_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_conv3d_8_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_conv3d_9_kernel"/device:CPU:0*
_output_shapes
 Д
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_conv3d_9_kernel^Read_6/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0z
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0**
_output_shapes
:z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_conv3d_9_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_conv3d_9_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_8/DisableCopyOnReadDisableCopyOnRead)read_8_disablecopyonread_conv3d_10_kernel"/device:CPU:0*
_output_shapes
 Е
Read_8/ReadVariableOpReadVariableOp)read_8_disablecopyonread_conv3d_10_kernel^Read_8/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0z
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0**
_output_shapes
:{
Read_9/DisableCopyOnReadDisableCopyOnRead'read_9_disablecopyonread_conv3d_10_bias"/device:CPU:0*
_output_shapes
 Ѓ
Read_9/ReadVariableOpReadVariableOp'read_9_disablecopyonread_conv3d_10_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_10/DisableCopyOnReadDisableCopyOnRead)read_10_disablecopyonread_conv3d_4_kernel"/device:CPU:0*
_output_shapes
 З
Read_10/ReadVariableOpReadVariableOp)read_10_disablecopyonread_conv3d_4_kernel^Read_10/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0{
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0**
_output_shapes
:|
Read_11/DisableCopyOnReadDisableCopyOnRead'read_11_disablecopyonread_conv3d_4_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_11/ReadVariableOpReadVariableOp'read_11_disablecopyonread_conv3d_4_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_12/DisableCopyOnReadDisableCopyOnRead)read_12_disablecopyonread_conv3d_5_kernel"/device:CPU:0*
_output_shapes
 З
Read_12/ReadVariableOpReadVariableOp)read_12_disablecopyonread_conv3d_5_kernel^Read_12/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0{
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0**
_output_shapes
:|
Read_13/DisableCopyOnReadDisableCopyOnRead'read_13_disablecopyonread_conv3d_5_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_13/ReadVariableOpReadVariableOp'read_13_disablecopyonread_conv3d_5_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_14/DisableCopyOnReadDisableCopyOnRead*read_14_disablecopyonread_conv3d_11_kernel"/device:CPU:0*
_output_shapes
 И
Read_14/ReadVariableOpReadVariableOp*read_14_disablecopyonread_conv3d_11_kernel^Read_14/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0{
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0**
_output_shapes
:}
Read_15/DisableCopyOnReadDisableCopyOnRead(read_15_disablecopyonread_conv3d_11_bias"/device:CPU:0*
_output_shapes
 І
Read_15/ReadVariableOpReadVariableOp(read_15_disablecopyonread_conv3d_11_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_16/DisableCopyOnReadDisableCopyOnRead*read_16_disablecopyonread_conv3d_12_kernel"/device:CPU:0*
_output_shapes
 И
Read_16/ReadVariableOpReadVariableOp*read_16_disablecopyonread_conv3d_12_kernel^Read_16/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0{
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0**
_output_shapes
:}
Read_17/DisableCopyOnReadDisableCopyOnRead(read_17_disablecopyonread_conv3d_12_bias"/device:CPU:0*
_output_shapes
 І
Read_17/ReadVariableOpReadVariableOp(read_17_disablecopyonread_conv3d_12_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_18/DisableCopyOnReadDisableCopyOnRead)read_18_disablecopyonread_conv3d_2_kernel"/device:CPU:0*
_output_shapes
 З
Read_18/ReadVariableOpReadVariableOp)read_18_disablecopyonread_conv3d_2_kernel^Read_18/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0{
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0**
_output_shapes
:|
Read_19/DisableCopyOnReadDisableCopyOnRead'read_19_disablecopyonread_conv3d_2_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_19/ReadVariableOpReadVariableOp'read_19_disablecopyonread_conv3d_2_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_20/DisableCopyOnReadDisableCopyOnRead)read_20_disablecopyonread_conv3d_3_kernel"/device:CPU:0*
_output_shapes
 З
Read_20/ReadVariableOpReadVariableOp)read_20_disablecopyonread_conv3d_3_kernel^Read_20/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0{
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0**
_output_shapes
:|
Read_21/DisableCopyOnReadDisableCopyOnRead'read_21_disablecopyonread_conv3d_3_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_21/ReadVariableOpReadVariableOp'read_21_disablecopyonread_conv3d_3_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_22/DisableCopyOnReadDisableCopyOnRead*read_22_disablecopyonread_conv3d_13_kernel"/device:CPU:0*
_output_shapes
 И
Read_22/ReadVariableOpReadVariableOp*read_22_disablecopyonread_conv3d_13_kernel^Read_22/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0{
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0**
_output_shapes
:}
Read_23/DisableCopyOnReadDisableCopyOnRead(read_23_disablecopyonread_conv3d_13_bias"/device:CPU:0*
_output_shapes
 І
Read_23/ReadVariableOpReadVariableOp(read_23_disablecopyonread_conv3d_13_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_24/DisableCopyOnReadDisableCopyOnRead*read_24_disablecopyonread_conv3d_14_kernel"/device:CPU:0*
_output_shapes
 И
Read_24/ReadVariableOpReadVariableOp*read_24_disablecopyonread_conv3d_14_kernel^Read_24/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0{
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0**
_output_shapes
:}
Read_25/DisableCopyOnReadDisableCopyOnRead(read_25_disablecopyonread_conv3d_14_bias"/device:CPU:0*
_output_shapes
 І
Read_25/ReadVariableOpReadVariableOp(read_25_disablecopyonread_conv3d_14_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_26/DisableCopyOnReadDisableCopyOnRead'read_26_disablecopyonread_conv3d_kernel"/device:CPU:0*
_output_shapes
 Е
Read_26/ReadVariableOpReadVariableOp'read_26_disablecopyonread_conv3d_kernel^Read_26/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0{
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0**
_output_shapes
:z
Read_27/DisableCopyOnReadDisableCopyOnRead%read_27_disablecopyonread_conv3d_bias"/device:CPU:0*
_output_shapes
 Ѓ
Read_27/ReadVariableOpReadVariableOp%read_27_disablecopyonread_conv3d_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_28/DisableCopyOnReadDisableCopyOnRead)read_28_disablecopyonread_conv3d_1_kernel"/device:CPU:0*
_output_shapes
 З
Read_28/ReadVariableOpReadVariableOp)read_28_disablecopyonread_conv3d_1_kernel^Read_28/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0{
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0**
_output_shapes
:|
Read_29/DisableCopyOnReadDisableCopyOnRead'read_29_disablecopyonread_conv3d_1_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_29/ReadVariableOpReadVariableOp'read_29_disablecopyonread_conv3d_1_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_30/DisableCopyOnReadDisableCopyOnRead*read_30_disablecopyonread_conv3d_15_kernel"/device:CPU:0*
_output_shapes
 И
Read_30/ReadVariableOpReadVariableOp*read_30_disablecopyonread_conv3d_15_kernel^Read_30/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0{
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0**
_output_shapes
:}
Read_31/DisableCopyOnReadDisableCopyOnRead(read_31_disablecopyonread_conv3d_15_bias"/device:CPU:0*
_output_shapes
 І
Read_31/ReadVariableOpReadVariableOp(read_31_disablecopyonread_conv3d_15_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_32/DisableCopyOnReadDisableCopyOnRead*read_32_disablecopyonread_conv3d_16_kernel"/device:CPU:0*
_output_shapes
 И
Read_32/ReadVariableOpReadVariableOp*read_32_disablecopyonread_conv3d_16_kernel^Read_32/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0{
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0**
_output_shapes
:}
Read_33/DisableCopyOnReadDisableCopyOnRead(read_33_disablecopyonread_conv3d_16_bias"/device:CPU:0*
_output_shapes
 І
Read_33/ReadVariableOpReadVariableOp(read_33_disablecopyonread_conv3d_16_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_34/DisableCopyOnReadDisableCopyOnRead*read_34_disablecopyonread_conv3d_17_kernel"/device:CPU:0*
_output_shapes
 И
Read_34/ReadVariableOpReadVariableOp*read_34_disablecopyonread_conv3d_17_kernel^Read_34/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0{
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0**
_output_shapes
:}
Read_35/DisableCopyOnReadDisableCopyOnRead(read_35_disablecopyonread_conv3d_17_bias"/device:CPU:0*
_output_shapes
 І
Read_35/ReadVariableOpReadVariableOp(read_35_disablecopyonread_conv3d_17_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_36/DisableCopyOnReadDisableCopyOnRead*read_36_disablecopyonread_conv3d_18_kernel"/device:CPU:0*
_output_shapes
 И
Read_36/ReadVariableOpReadVariableOp*read_36_disablecopyonread_conv3d_18_kernel^Read_36/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0{
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0**
_output_shapes
:}
Read_37/DisableCopyOnReadDisableCopyOnRead(read_37_disablecopyonread_conv3d_18_bias"/device:CPU:0*
_output_shapes
 І
Read_37/ReadVariableOpReadVariableOp(read_37_disablecopyonread_conv3d_18_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_38/DisableCopyOnReadDisableCopyOnRead#read_38_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 
Read_38/ReadVariableOpReadVariableOp#read_38_disablecopyonread_iteration^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_39/DisableCopyOnReadDisableCopyOnRead'read_39_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 Ё
Read_39/ReadVariableOpReadVariableOp'read_39_disablecopyonread_learning_rate^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_40/DisableCopyOnReadDisableCopyOnRead0read_40_disablecopyonread_adam_m_conv3d_6_kernel"/device:CPU:0*
_output_shapes
 О
Read_40/ReadVariableOpReadVariableOp0read_40_disablecopyonread_adam_m_conv3d_6_kernel^Read_40/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0{
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_41/DisableCopyOnReadDisableCopyOnRead0read_41_disablecopyonread_adam_v_conv3d_6_kernel"/device:CPU:0*
_output_shapes
 О
Read_41/ReadVariableOpReadVariableOp0read_41_disablecopyonread_adam_v_conv3d_6_kernel^Read_41/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0{
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_42/DisableCopyOnReadDisableCopyOnRead.read_42_disablecopyonread_adam_m_conv3d_6_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_42/ReadVariableOpReadVariableOp.read_42_disablecopyonread_adam_m_conv3d_6_bias^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_43/DisableCopyOnReadDisableCopyOnRead.read_43_disablecopyonread_adam_v_conv3d_6_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_43/ReadVariableOpReadVariableOp.read_43_disablecopyonread_adam_v_conv3d_6_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_44/DisableCopyOnReadDisableCopyOnRead0read_44_disablecopyonread_adam_m_conv3d_7_kernel"/device:CPU:0*
_output_shapes
 О
Read_44/ReadVariableOpReadVariableOp0read_44_disablecopyonread_adam_m_conv3d_7_kernel^Read_44/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0{
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_45/DisableCopyOnReadDisableCopyOnRead0read_45_disablecopyonread_adam_v_conv3d_7_kernel"/device:CPU:0*
_output_shapes
 О
Read_45/ReadVariableOpReadVariableOp0read_45_disablecopyonread_adam_v_conv3d_7_kernel^Read_45/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0{
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_46/DisableCopyOnReadDisableCopyOnRead.read_46_disablecopyonread_adam_m_conv3d_7_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_46/ReadVariableOpReadVariableOp.read_46_disablecopyonread_adam_m_conv3d_7_bias^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_47/DisableCopyOnReadDisableCopyOnRead.read_47_disablecopyonread_adam_v_conv3d_7_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_47/ReadVariableOpReadVariableOp.read_47_disablecopyonread_adam_v_conv3d_7_bias^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_48/DisableCopyOnReadDisableCopyOnRead0read_48_disablecopyonread_adam_m_conv3d_8_kernel"/device:CPU:0*
_output_shapes
 О
Read_48/ReadVariableOpReadVariableOp0read_48_disablecopyonread_adam_m_conv3d_8_kernel^Read_48/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0{
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_49/DisableCopyOnReadDisableCopyOnRead0read_49_disablecopyonread_adam_v_conv3d_8_kernel"/device:CPU:0*
_output_shapes
 О
Read_49/ReadVariableOpReadVariableOp0read_49_disablecopyonread_adam_v_conv3d_8_kernel^Read_49/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0{
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:q
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_50/DisableCopyOnReadDisableCopyOnRead.read_50_disablecopyonread_adam_m_conv3d_8_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_50/ReadVariableOpReadVariableOp.read_50_disablecopyonread_adam_m_conv3d_8_bias^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_51/DisableCopyOnReadDisableCopyOnRead.read_51_disablecopyonread_adam_v_conv3d_8_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_51/ReadVariableOpReadVariableOp.read_51_disablecopyonread_adam_v_conv3d_8_bias^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_52/DisableCopyOnReadDisableCopyOnRead0read_52_disablecopyonread_adam_m_conv3d_9_kernel"/device:CPU:0*
_output_shapes
 О
Read_52/ReadVariableOpReadVariableOp0read_52_disablecopyonread_adam_m_conv3d_9_kernel^Read_52/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_53/DisableCopyOnReadDisableCopyOnRead0read_53_disablecopyonread_adam_v_conv3d_9_kernel"/device:CPU:0*
_output_shapes
 О
Read_53/ReadVariableOpReadVariableOp0read_53_disablecopyonread_adam_v_conv3d_9_kernel^Read_53/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_54/DisableCopyOnReadDisableCopyOnRead.read_54_disablecopyonread_adam_m_conv3d_9_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_54/ReadVariableOpReadVariableOp.read_54_disablecopyonread_adam_m_conv3d_9_bias^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_55/DisableCopyOnReadDisableCopyOnRead.read_55_disablecopyonread_adam_v_conv3d_9_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_55/ReadVariableOpReadVariableOp.read_55_disablecopyonread_adam_v_conv3d_9_bias^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_56/DisableCopyOnReadDisableCopyOnRead1read_56_disablecopyonread_adam_m_conv3d_10_kernel"/device:CPU:0*
_output_shapes
 П
Read_56/ReadVariableOpReadVariableOp1read_56_disablecopyonread_adam_m_conv3d_10_kernel^Read_56/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_57/DisableCopyOnReadDisableCopyOnRead1read_57_disablecopyonread_adam_v_conv3d_10_kernel"/device:CPU:0*
_output_shapes
 П
Read_57/ReadVariableOpReadVariableOp1read_57_disablecopyonread_adam_v_conv3d_10_kernel^Read_57/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_58/DisableCopyOnReadDisableCopyOnRead/read_58_disablecopyonread_adam_m_conv3d_10_bias"/device:CPU:0*
_output_shapes
 ­
Read_58/ReadVariableOpReadVariableOp/read_58_disablecopyonread_adam_m_conv3d_10_bias^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_59/DisableCopyOnReadDisableCopyOnRead/read_59_disablecopyonread_adam_v_conv3d_10_bias"/device:CPU:0*
_output_shapes
 ­
Read_59/ReadVariableOpReadVariableOp/read_59_disablecopyonread_adam_v_conv3d_10_bias^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_60/DisableCopyOnReadDisableCopyOnRead0read_60_disablecopyonread_adam_m_conv3d_4_kernel"/device:CPU:0*
_output_shapes
 О
Read_60/ReadVariableOpReadVariableOp0read_60_disablecopyonread_adam_m_conv3d_4_kernel^Read_60/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_61/DisableCopyOnReadDisableCopyOnRead0read_61_disablecopyonread_adam_v_conv3d_4_kernel"/device:CPU:0*
_output_shapes
 О
Read_61/ReadVariableOpReadVariableOp0read_61_disablecopyonread_adam_v_conv3d_4_kernel^Read_61/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_62/DisableCopyOnReadDisableCopyOnRead.read_62_disablecopyonread_adam_m_conv3d_4_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_62/ReadVariableOpReadVariableOp.read_62_disablecopyonread_adam_m_conv3d_4_bias^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_63/DisableCopyOnReadDisableCopyOnRead.read_63_disablecopyonread_adam_v_conv3d_4_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_63/ReadVariableOpReadVariableOp.read_63_disablecopyonread_adam_v_conv3d_4_bias^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_64/DisableCopyOnReadDisableCopyOnRead0read_64_disablecopyonread_adam_m_conv3d_5_kernel"/device:CPU:0*
_output_shapes
 О
Read_64/ReadVariableOpReadVariableOp0read_64_disablecopyonread_adam_m_conv3d_5_kernel^Read_64/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_65/DisableCopyOnReadDisableCopyOnRead0read_65_disablecopyonread_adam_v_conv3d_5_kernel"/device:CPU:0*
_output_shapes
 О
Read_65/ReadVariableOpReadVariableOp0read_65_disablecopyonread_adam_v_conv3d_5_kernel^Read_65/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_66/DisableCopyOnReadDisableCopyOnRead.read_66_disablecopyonread_adam_m_conv3d_5_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_66/ReadVariableOpReadVariableOp.read_66_disablecopyonread_adam_m_conv3d_5_bias^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_67/DisableCopyOnReadDisableCopyOnRead.read_67_disablecopyonread_adam_v_conv3d_5_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_67/ReadVariableOpReadVariableOp.read_67_disablecopyonread_adam_v_conv3d_5_bias^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_68/DisableCopyOnReadDisableCopyOnRead1read_68_disablecopyonread_adam_m_conv3d_11_kernel"/device:CPU:0*
_output_shapes
 П
Read_68/ReadVariableOpReadVariableOp1read_68_disablecopyonread_adam_m_conv3d_11_kernel^Read_68/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_69/DisableCopyOnReadDisableCopyOnRead1read_69_disablecopyonread_adam_v_conv3d_11_kernel"/device:CPU:0*
_output_shapes
 П
Read_69/ReadVariableOpReadVariableOp1read_69_disablecopyonread_adam_v_conv3d_11_kernel^Read_69/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_70/DisableCopyOnReadDisableCopyOnRead/read_70_disablecopyonread_adam_m_conv3d_11_bias"/device:CPU:0*
_output_shapes
 ­
Read_70/ReadVariableOpReadVariableOp/read_70_disablecopyonread_adam_m_conv3d_11_bias^Read_70/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_71/DisableCopyOnReadDisableCopyOnRead/read_71_disablecopyonread_adam_v_conv3d_11_bias"/device:CPU:0*
_output_shapes
 ­
Read_71/ReadVariableOpReadVariableOp/read_71_disablecopyonread_adam_v_conv3d_11_bias^Read_71/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_72/DisableCopyOnReadDisableCopyOnRead1read_72_disablecopyonread_adam_m_conv3d_12_kernel"/device:CPU:0*
_output_shapes
 П
Read_72/ReadVariableOpReadVariableOp1read_72_disablecopyonread_adam_m_conv3d_12_kernel^Read_72/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_73/DisableCopyOnReadDisableCopyOnRead1read_73_disablecopyonread_adam_v_conv3d_12_kernel"/device:CPU:0*
_output_shapes
 П
Read_73/ReadVariableOpReadVariableOp1read_73_disablecopyonread_adam_v_conv3d_12_kernel^Read_73/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_74/DisableCopyOnReadDisableCopyOnRead/read_74_disablecopyonread_adam_m_conv3d_12_bias"/device:CPU:0*
_output_shapes
 ­
Read_74/ReadVariableOpReadVariableOp/read_74_disablecopyonread_adam_m_conv3d_12_bias^Read_74/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_75/DisableCopyOnReadDisableCopyOnRead/read_75_disablecopyonread_adam_v_conv3d_12_bias"/device:CPU:0*
_output_shapes
 ­
Read_75/ReadVariableOpReadVariableOp/read_75_disablecopyonread_adam_v_conv3d_12_bias^Read_75/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_76/DisableCopyOnReadDisableCopyOnRead0read_76_disablecopyonread_adam_m_conv3d_2_kernel"/device:CPU:0*
_output_shapes
 О
Read_76/ReadVariableOpReadVariableOp0read_76_disablecopyonread_adam_m_conv3d_2_kernel^Read_76/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_152IdentityRead_76/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_77/DisableCopyOnReadDisableCopyOnRead0read_77_disablecopyonread_adam_v_conv3d_2_kernel"/device:CPU:0*
_output_shapes
 О
Read_77/ReadVariableOpReadVariableOp0read_77_disablecopyonread_adam_v_conv3d_2_kernel^Read_77/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_154IdentityRead_77/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_78/DisableCopyOnReadDisableCopyOnRead.read_78_disablecopyonread_adam_m_conv3d_2_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_78/ReadVariableOpReadVariableOp.read_78_disablecopyonread_adam_m_conv3d_2_bias^Read_78/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_156IdentityRead_78/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_79/DisableCopyOnReadDisableCopyOnRead.read_79_disablecopyonread_adam_v_conv3d_2_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_79/ReadVariableOpReadVariableOp.read_79_disablecopyonread_adam_v_conv3d_2_bias^Read_79/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_158IdentityRead_79/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_159IdentityIdentity_158:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_80/DisableCopyOnReadDisableCopyOnRead0read_80_disablecopyonread_adam_m_conv3d_3_kernel"/device:CPU:0*
_output_shapes
 О
Read_80/ReadVariableOpReadVariableOp0read_80_disablecopyonread_adam_m_conv3d_3_kernel^Read_80/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_160IdentityRead_80/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_161IdentityIdentity_160:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_81/DisableCopyOnReadDisableCopyOnRead0read_81_disablecopyonread_adam_v_conv3d_3_kernel"/device:CPU:0*
_output_shapes
 О
Read_81/ReadVariableOpReadVariableOp0read_81_disablecopyonread_adam_v_conv3d_3_kernel^Read_81/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_162IdentityRead_81/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_163IdentityIdentity_162:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_82/DisableCopyOnReadDisableCopyOnRead.read_82_disablecopyonread_adam_m_conv3d_3_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_82/ReadVariableOpReadVariableOp.read_82_disablecopyonread_adam_m_conv3d_3_bias^Read_82/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_164IdentityRead_82/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_165IdentityIdentity_164:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_83/DisableCopyOnReadDisableCopyOnRead.read_83_disablecopyonread_adam_v_conv3d_3_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_83/ReadVariableOpReadVariableOp.read_83_disablecopyonread_adam_v_conv3d_3_bias^Read_83/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_166IdentityRead_83/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_167IdentityIdentity_166:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_84/DisableCopyOnReadDisableCopyOnRead1read_84_disablecopyonread_adam_m_conv3d_13_kernel"/device:CPU:0*
_output_shapes
 П
Read_84/ReadVariableOpReadVariableOp1read_84_disablecopyonread_adam_m_conv3d_13_kernel^Read_84/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_168IdentityRead_84/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_169IdentityIdentity_168:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_85/DisableCopyOnReadDisableCopyOnRead1read_85_disablecopyonread_adam_v_conv3d_13_kernel"/device:CPU:0*
_output_shapes
 П
Read_85/ReadVariableOpReadVariableOp1read_85_disablecopyonread_adam_v_conv3d_13_kernel^Read_85/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_170IdentityRead_85/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_171IdentityIdentity_170:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_86/DisableCopyOnReadDisableCopyOnRead/read_86_disablecopyonread_adam_m_conv3d_13_bias"/device:CPU:0*
_output_shapes
 ­
Read_86/ReadVariableOpReadVariableOp/read_86_disablecopyonread_adam_m_conv3d_13_bias^Read_86/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_172IdentityRead_86/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_173IdentityIdentity_172:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_87/DisableCopyOnReadDisableCopyOnRead/read_87_disablecopyonread_adam_v_conv3d_13_bias"/device:CPU:0*
_output_shapes
 ­
Read_87/ReadVariableOpReadVariableOp/read_87_disablecopyonread_adam_v_conv3d_13_bias^Read_87/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_174IdentityRead_87/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_175IdentityIdentity_174:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_88/DisableCopyOnReadDisableCopyOnRead1read_88_disablecopyonread_adam_m_conv3d_14_kernel"/device:CPU:0*
_output_shapes
 П
Read_88/ReadVariableOpReadVariableOp1read_88_disablecopyonread_adam_m_conv3d_14_kernel^Read_88/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_176IdentityRead_88/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_177IdentityIdentity_176:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_89/DisableCopyOnReadDisableCopyOnRead1read_89_disablecopyonread_adam_v_conv3d_14_kernel"/device:CPU:0*
_output_shapes
 П
Read_89/ReadVariableOpReadVariableOp1read_89_disablecopyonread_adam_v_conv3d_14_kernel^Read_89/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_178IdentityRead_89/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_179IdentityIdentity_178:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_90/DisableCopyOnReadDisableCopyOnRead/read_90_disablecopyonread_adam_m_conv3d_14_bias"/device:CPU:0*
_output_shapes
 ­
Read_90/ReadVariableOpReadVariableOp/read_90_disablecopyonread_adam_m_conv3d_14_bias^Read_90/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_180IdentityRead_90/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_181IdentityIdentity_180:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_91/DisableCopyOnReadDisableCopyOnRead/read_91_disablecopyonread_adam_v_conv3d_14_bias"/device:CPU:0*
_output_shapes
 ­
Read_91/ReadVariableOpReadVariableOp/read_91_disablecopyonread_adam_v_conv3d_14_bias^Read_91/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_182IdentityRead_91/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_183IdentityIdentity_182:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_92/DisableCopyOnReadDisableCopyOnRead.read_92_disablecopyonread_adam_m_conv3d_kernel"/device:CPU:0*
_output_shapes
 М
Read_92/ReadVariableOpReadVariableOp.read_92_disablecopyonread_adam_m_conv3d_kernel^Read_92/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_184IdentityRead_92/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_185IdentityIdentity_184:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_93/DisableCopyOnReadDisableCopyOnRead.read_93_disablecopyonread_adam_v_conv3d_kernel"/device:CPU:0*
_output_shapes
 М
Read_93/ReadVariableOpReadVariableOp.read_93_disablecopyonread_adam_v_conv3d_kernel^Read_93/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_186IdentityRead_93/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_187IdentityIdentity_186:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_94/DisableCopyOnReadDisableCopyOnRead,read_94_disablecopyonread_adam_m_conv3d_bias"/device:CPU:0*
_output_shapes
 Њ
Read_94/ReadVariableOpReadVariableOp,read_94_disablecopyonread_adam_m_conv3d_bias^Read_94/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_188IdentityRead_94/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_189IdentityIdentity_188:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_95/DisableCopyOnReadDisableCopyOnRead,read_95_disablecopyonread_adam_v_conv3d_bias"/device:CPU:0*
_output_shapes
 Њ
Read_95/ReadVariableOpReadVariableOp,read_95_disablecopyonread_adam_v_conv3d_bias^Read_95/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_190IdentityRead_95/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_191IdentityIdentity_190:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_96/DisableCopyOnReadDisableCopyOnRead0read_96_disablecopyonread_adam_m_conv3d_1_kernel"/device:CPU:0*
_output_shapes
 О
Read_96/ReadVariableOpReadVariableOp0read_96_disablecopyonread_adam_m_conv3d_1_kernel^Read_96/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_192IdentityRead_96/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_193IdentityIdentity_192:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_97/DisableCopyOnReadDisableCopyOnRead0read_97_disablecopyonread_adam_v_conv3d_1_kernel"/device:CPU:0*
_output_shapes
 О
Read_97/ReadVariableOpReadVariableOp0read_97_disablecopyonread_adam_v_conv3d_1_kernel^Read_97/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0|
Identity_194IdentityRead_97/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_195IdentityIdentity_194:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_98/DisableCopyOnReadDisableCopyOnRead.read_98_disablecopyonread_adam_m_conv3d_1_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_98/ReadVariableOpReadVariableOp.read_98_disablecopyonread_adam_m_conv3d_1_bias^Read_98/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_196IdentityRead_98/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_197IdentityIdentity_196:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_99/DisableCopyOnReadDisableCopyOnRead.read_99_disablecopyonread_adam_v_conv3d_1_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_99/ReadVariableOpReadVariableOp.read_99_disablecopyonread_adam_v_conv3d_1_bias^Read_99/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_198IdentityRead_99/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_199IdentityIdentity_198:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_100/DisableCopyOnReadDisableCopyOnRead2read_100_disablecopyonread_adam_m_conv3d_15_kernel"/device:CPU:0*
_output_shapes
 Т
Read_100/ReadVariableOpReadVariableOp2read_100_disablecopyonread_adam_m_conv3d_15_kernel^Read_100/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0}
Identity_200IdentityRead_100/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_201IdentityIdentity_200:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_101/DisableCopyOnReadDisableCopyOnRead2read_101_disablecopyonread_adam_v_conv3d_15_kernel"/device:CPU:0*
_output_shapes
 Т
Read_101/ReadVariableOpReadVariableOp2read_101_disablecopyonread_adam_v_conv3d_15_kernel^Read_101/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0}
Identity_202IdentityRead_101/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_203IdentityIdentity_202:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_102/DisableCopyOnReadDisableCopyOnRead0read_102_disablecopyonread_adam_m_conv3d_15_bias"/device:CPU:0*
_output_shapes
 А
Read_102/ReadVariableOpReadVariableOp0read_102_disablecopyonread_adam_m_conv3d_15_bias^Read_102/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_204IdentityRead_102/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_205IdentityIdentity_204:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_103/DisableCopyOnReadDisableCopyOnRead0read_103_disablecopyonread_adam_v_conv3d_15_bias"/device:CPU:0*
_output_shapes
 А
Read_103/ReadVariableOpReadVariableOp0read_103_disablecopyonread_adam_v_conv3d_15_bias^Read_103/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_206IdentityRead_103/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_207IdentityIdentity_206:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_104/DisableCopyOnReadDisableCopyOnRead2read_104_disablecopyonread_adam_m_conv3d_16_kernel"/device:CPU:0*
_output_shapes
 Т
Read_104/ReadVariableOpReadVariableOp2read_104_disablecopyonread_adam_m_conv3d_16_kernel^Read_104/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0}
Identity_208IdentityRead_104/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_209IdentityIdentity_208:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_105/DisableCopyOnReadDisableCopyOnRead2read_105_disablecopyonread_adam_v_conv3d_16_kernel"/device:CPU:0*
_output_shapes
 Т
Read_105/ReadVariableOpReadVariableOp2read_105_disablecopyonread_adam_v_conv3d_16_kernel^Read_105/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0}
Identity_210IdentityRead_105/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_211IdentityIdentity_210:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_106/DisableCopyOnReadDisableCopyOnRead0read_106_disablecopyonread_adam_m_conv3d_16_bias"/device:CPU:0*
_output_shapes
 А
Read_106/ReadVariableOpReadVariableOp0read_106_disablecopyonread_adam_m_conv3d_16_bias^Read_106/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_212IdentityRead_106/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_213IdentityIdentity_212:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_107/DisableCopyOnReadDisableCopyOnRead0read_107_disablecopyonread_adam_v_conv3d_16_bias"/device:CPU:0*
_output_shapes
 А
Read_107/ReadVariableOpReadVariableOp0read_107_disablecopyonread_adam_v_conv3d_16_bias^Read_107/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_214IdentityRead_107/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_215IdentityIdentity_214:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_108/DisableCopyOnReadDisableCopyOnRead2read_108_disablecopyonread_adam_m_conv3d_17_kernel"/device:CPU:0*
_output_shapes
 Т
Read_108/ReadVariableOpReadVariableOp2read_108_disablecopyonread_adam_m_conv3d_17_kernel^Read_108/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0}
Identity_216IdentityRead_108/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_217IdentityIdentity_216:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_109/DisableCopyOnReadDisableCopyOnRead2read_109_disablecopyonread_adam_v_conv3d_17_kernel"/device:CPU:0*
_output_shapes
 Т
Read_109/ReadVariableOpReadVariableOp2read_109_disablecopyonread_adam_v_conv3d_17_kernel^Read_109/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0}
Identity_218IdentityRead_109/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_219IdentityIdentity_218:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_110/DisableCopyOnReadDisableCopyOnRead0read_110_disablecopyonread_adam_m_conv3d_17_bias"/device:CPU:0*
_output_shapes
 А
Read_110/ReadVariableOpReadVariableOp0read_110_disablecopyonread_adam_m_conv3d_17_bias^Read_110/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_220IdentityRead_110/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_221IdentityIdentity_220:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_111/DisableCopyOnReadDisableCopyOnRead0read_111_disablecopyonread_adam_v_conv3d_17_bias"/device:CPU:0*
_output_shapes
 А
Read_111/ReadVariableOpReadVariableOp0read_111_disablecopyonread_adam_v_conv3d_17_bias^Read_111/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_222IdentityRead_111/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_223IdentityIdentity_222:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_112/DisableCopyOnReadDisableCopyOnRead2read_112_disablecopyonread_adam_m_conv3d_18_kernel"/device:CPU:0*
_output_shapes
 Т
Read_112/ReadVariableOpReadVariableOp2read_112_disablecopyonread_adam_m_conv3d_18_kernel^Read_112/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0}
Identity_224IdentityRead_112/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_225IdentityIdentity_224:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_113/DisableCopyOnReadDisableCopyOnRead2read_113_disablecopyonread_adam_v_conv3d_18_kernel"/device:CPU:0*
_output_shapes
 Т
Read_113/ReadVariableOpReadVariableOp2read_113_disablecopyonread_adam_v_conv3d_18_kernel^Read_113/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:*
dtype0}
Identity_226IdentityRead_113/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:s
Identity_227IdentityIdentity_226:output:0"/device:CPU:0*
T0**
_output_shapes
:
Read_114/DisableCopyOnReadDisableCopyOnRead0read_114_disablecopyonread_adam_m_conv3d_18_bias"/device:CPU:0*
_output_shapes
 А
Read_114/ReadVariableOpReadVariableOp0read_114_disablecopyonread_adam_m_conv3d_18_bias^Read_114/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_228IdentityRead_114/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_229IdentityIdentity_228:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_115/DisableCopyOnReadDisableCopyOnRead0read_115_disablecopyonread_adam_v_conv3d_18_bias"/device:CPU:0*
_output_shapes
 А
Read_115/ReadVariableOpReadVariableOp0read_115_disablecopyonread_adam_v_conv3d_18_bias^Read_115/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_230IdentityRead_115/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_231IdentityIdentity_230:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_116/DisableCopyOnReadDisableCopyOnRead read_116_disablecopyonread_total"/device:CPU:0*
_output_shapes
 
Read_116/ReadVariableOpReadVariableOp read_116_disablecopyonread_total^Read_116/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_232IdentityRead_116/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_233IdentityIdentity_232:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_117/DisableCopyOnReadDisableCopyOnRead read_117_disablecopyonread_count"/device:CPU:0*
_output_shapes
 
Read_117/ReadVariableOpReadVariableOp read_117_disablecopyonread_count^Read_117/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_234IdentityRead_117/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_235IdentityIdentity_234:output:0"/device:CPU:0*
T0*
_output_shapes
: 2
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:w*
dtype0*Љ1
value1B1wB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/73/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/74/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/75/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/76/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHо
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:w*
dtype0*
valueљBіwB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B В
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0Identity_159:output:0Identity_161:output:0Identity_163:output:0Identity_165:output:0Identity_167:output:0Identity_169:output:0Identity_171:output:0Identity_173:output:0Identity_175:output:0Identity_177:output:0Identity_179:output:0Identity_181:output:0Identity_183:output:0Identity_185:output:0Identity_187:output:0Identity_189:output:0Identity_191:output:0Identity_193:output:0Identity_195:output:0Identity_197:output:0Identity_199:output:0Identity_201:output:0Identity_203:output:0Identity_205:output:0Identity_207:output:0Identity_209:output:0Identity_211:output:0Identity_213:output:0Identity_215:output:0Identity_217:output:0Identity_219:output:0Identity_221:output:0Identity_223:output:0Identity_225:output:0Identity_227:output:0Identity_229:output:0Identity_231:output:0Identity_233:output:0Identity_235:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes{
y2w	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_236Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_237IdentityIdentity_236:output:0^NoOp*
T0*
_output_shapes
: Б1
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_100/DisableCopyOnRead^Read_100/ReadVariableOp^Read_101/DisableCopyOnRead^Read_101/ReadVariableOp^Read_102/DisableCopyOnRead^Read_102/ReadVariableOp^Read_103/DisableCopyOnRead^Read_103/ReadVariableOp^Read_104/DisableCopyOnRead^Read_104/ReadVariableOp^Read_105/DisableCopyOnRead^Read_105/ReadVariableOp^Read_106/DisableCopyOnRead^Read_106/ReadVariableOp^Read_107/DisableCopyOnRead^Read_107/ReadVariableOp^Read_108/DisableCopyOnRead^Read_108/ReadVariableOp^Read_109/DisableCopyOnRead^Read_109/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_110/DisableCopyOnRead^Read_110/ReadVariableOp^Read_111/DisableCopyOnRead^Read_111/ReadVariableOp^Read_112/DisableCopyOnRead^Read_112/ReadVariableOp^Read_113/DisableCopyOnRead^Read_113/ReadVariableOp^Read_114/DisableCopyOnRead^Read_114/ReadVariableOp^Read_115/DisableCopyOnRead^Read_115/ReadVariableOp^Read_116/DisableCopyOnRead^Read_116/ReadVariableOp^Read_117/DisableCopyOnRead^Read_117/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_79/DisableCopyOnRead^Read_79/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_80/DisableCopyOnRead^Read_80/ReadVariableOp^Read_81/DisableCopyOnRead^Read_81/ReadVariableOp^Read_82/DisableCopyOnRead^Read_82/ReadVariableOp^Read_83/DisableCopyOnRead^Read_83/ReadVariableOp^Read_84/DisableCopyOnRead^Read_84/ReadVariableOp^Read_85/DisableCopyOnRead^Read_85/ReadVariableOp^Read_86/DisableCopyOnRead^Read_86/ReadVariableOp^Read_87/DisableCopyOnRead^Read_87/ReadVariableOp^Read_88/DisableCopyOnRead^Read_88/ReadVariableOp^Read_89/DisableCopyOnRead^Read_89/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp^Read_90/DisableCopyOnRead^Read_90/ReadVariableOp^Read_91/DisableCopyOnRead^Read_91/ReadVariableOp^Read_92/DisableCopyOnRead^Read_92/ReadVariableOp^Read_93/DisableCopyOnRead^Read_93/ReadVariableOp^Read_94/DisableCopyOnRead^Read_94/ReadVariableOp^Read_95/DisableCopyOnRead^Read_95/ReadVariableOp^Read_96/DisableCopyOnRead^Read_96/ReadVariableOp^Read_97/DisableCopyOnRead^Read_97/ReadVariableOp^Read_98/DisableCopyOnRead^Read_98/ReadVariableOp^Read_99/DisableCopyOnRead^Read_99/ReadVariableOp*
_output_shapes
 "%
identity_237Identity_237:output:0*(
_construction_contextkEagerRuntime*
_input_shapesѓ
№: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp28
Read_100/DisableCopyOnReadRead_100/DisableCopyOnRead22
Read_100/ReadVariableOpRead_100/ReadVariableOp28
Read_101/DisableCopyOnReadRead_101/DisableCopyOnRead22
Read_101/ReadVariableOpRead_101/ReadVariableOp28
Read_102/DisableCopyOnReadRead_102/DisableCopyOnRead22
Read_102/ReadVariableOpRead_102/ReadVariableOp28
Read_103/DisableCopyOnReadRead_103/DisableCopyOnRead22
Read_103/ReadVariableOpRead_103/ReadVariableOp28
Read_104/DisableCopyOnReadRead_104/DisableCopyOnRead22
Read_104/ReadVariableOpRead_104/ReadVariableOp28
Read_105/DisableCopyOnReadRead_105/DisableCopyOnRead22
Read_105/ReadVariableOpRead_105/ReadVariableOp28
Read_106/DisableCopyOnReadRead_106/DisableCopyOnRead22
Read_106/ReadVariableOpRead_106/ReadVariableOp28
Read_107/DisableCopyOnReadRead_107/DisableCopyOnRead22
Read_107/ReadVariableOpRead_107/ReadVariableOp28
Read_108/DisableCopyOnReadRead_108/DisableCopyOnRead22
Read_108/ReadVariableOpRead_108/ReadVariableOp28
Read_109/DisableCopyOnReadRead_109/DisableCopyOnRead22
Read_109/ReadVariableOpRead_109/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp28
Read_110/DisableCopyOnReadRead_110/DisableCopyOnRead22
Read_110/ReadVariableOpRead_110/ReadVariableOp28
Read_111/DisableCopyOnReadRead_111/DisableCopyOnRead22
Read_111/ReadVariableOpRead_111/ReadVariableOp28
Read_112/DisableCopyOnReadRead_112/DisableCopyOnRead22
Read_112/ReadVariableOpRead_112/ReadVariableOp28
Read_113/DisableCopyOnReadRead_113/DisableCopyOnRead22
Read_113/ReadVariableOpRead_113/ReadVariableOp28
Read_114/DisableCopyOnReadRead_114/DisableCopyOnRead22
Read_114/ReadVariableOpRead_114/ReadVariableOp28
Read_115/DisableCopyOnReadRead_115/DisableCopyOnRead22
Read_115/ReadVariableOpRead_115/ReadVariableOp28
Read_116/DisableCopyOnReadRead_116/DisableCopyOnRead22
Read_116/ReadVariableOpRead_116/ReadVariableOp28
Read_117/DisableCopyOnReadRead_117/DisableCopyOnRead22
Read_117/ReadVariableOpRead_117/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp26
Read_79/DisableCopyOnReadRead_79/DisableCopyOnRead20
Read_79/ReadVariableOpRead_79/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp26
Read_80/DisableCopyOnReadRead_80/DisableCopyOnRead20
Read_80/ReadVariableOpRead_80/ReadVariableOp26
Read_81/DisableCopyOnReadRead_81/DisableCopyOnRead20
Read_81/ReadVariableOpRead_81/ReadVariableOp26
Read_82/DisableCopyOnReadRead_82/DisableCopyOnRead20
Read_82/ReadVariableOpRead_82/ReadVariableOp26
Read_83/DisableCopyOnReadRead_83/DisableCopyOnRead20
Read_83/ReadVariableOpRead_83/ReadVariableOp26
Read_84/DisableCopyOnReadRead_84/DisableCopyOnRead20
Read_84/ReadVariableOpRead_84/ReadVariableOp26
Read_85/DisableCopyOnReadRead_85/DisableCopyOnRead20
Read_85/ReadVariableOpRead_85/ReadVariableOp26
Read_86/DisableCopyOnReadRead_86/DisableCopyOnRead20
Read_86/ReadVariableOpRead_86/ReadVariableOp26
Read_87/DisableCopyOnReadRead_87/DisableCopyOnRead20
Read_87/ReadVariableOpRead_87/ReadVariableOp26
Read_88/DisableCopyOnReadRead_88/DisableCopyOnRead20
Read_88/ReadVariableOpRead_88/ReadVariableOp26
Read_89/DisableCopyOnReadRead_89/DisableCopyOnRead20
Read_89/ReadVariableOpRead_89/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp26
Read_90/DisableCopyOnReadRead_90/DisableCopyOnRead20
Read_90/ReadVariableOpRead_90/ReadVariableOp26
Read_91/DisableCopyOnReadRead_91/DisableCopyOnRead20
Read_91/ReadVariableOpRead_91/ReadVariableOp26
Read_92/DisableCopyOnReadRead_92/DisableCopyOnRead20
Read_92/ReadVariableOpRead_92/ReadVariableOp26
Read_93/DisableCopyOnReadRead_93/DisableCopyOnRead20
Read_93/ReadVariableOpRead_93/ReadVariableOp26
Read_94/DisableCopyOnReadRead_94/DisableCopyOnRead20
Read_94/ReadVariableOpRead_94/ReadVariableOp26
Read_95/DisableCopyOnReadRead_95/DisableCopyOnRead20
Read_95/ReadVariableOpRead_95/ReadVariableOp26
Read_96/DisableCopyOnReadRead_96/DisableCopyOnRead20
Read_96/ReadVariableOpRead_96/ReadVariableOp26
Read_97/DisableCopyOnReadRead_97/DisableCopyOnRead20
Read_97/ReadVariableOpRead_97/ReadVariableOp26
Read_98/DisableCopyOnReadRead_98/DisableCopyOnRead20
Read_98/ReadVariableOpRead_98/ReadVariableOp26
Read_99/DisableCopyOnReadRead_99/DisableCopyOnRead20
Read_99/ReadVariableOpRead_99/ReadVariableOp:=w9

_output_shapes
: 

_user_specified_nameConst:%v!

_user_specified_namecount:%u!

_user_specified_nametotal:5t1
/
_user_specified_nameAdam/v/conv3d_18/bias:5s1
/
_user_specified_nameAdam/m/conv3d_18/bias:7r3
1
_user_specified_nameAdam/v/conv3d_18/kernel:7q3
1
_user_specified_nameAdam/m/conv3d_18/kernel:5p1
/
_user_specified_nameAdam/v/conv3d_17/bias:5o1
/
_user_specified_nameAdam/m/conv3d_17/bias:7n3
1
_user_specified_nameAdam/v/conv3d_17/kernel:7m3
1
_user_specified_nameAdam/m/conv3d_17/kernel:5l1
/
_user_specified_nameAdam/v/conv3d_16/bias:5k1
/
_user_specified_nameAdam/m/conv3d_16/bias:7j3
1
_user_specified_nameAdam/v/conv3d_16/kernel:7i3
1
_user_specified_nameAdam/m/conv3d_16/kernel:5h1
/
_user_specified_nameAdam/v/conv3d_15/bias:5g1
/
_user_specified_nameAdam/m/conv3d_15/bias:7f3
1
_user_specified_nameAdam/v/conv3d_15/kernel:7e3
1
_user_specified_nameAdam/m/conv3d_15/kernel:4d0
.
_user_specified_nameAdam/v/conv3d_1/bias:4c0
.
_user_specified_nameAdam/m/conv3d_1/bias:6b2
0
_user_specified_nameAdam/v/conv3d_1/kernel:6a2
0
_user_specified_nameAdam/m/conv3d_1/kernel:2`.
,
_user_specified_nameAdam/v/conv3d/bias:2_.
,
_user_specified_nameAdam/m/conv3d/bias:4^0
.
_user_specified_nameAdam/v/conv3d/kernel:4]0
.
_user_specified_nameAdam/m/conv3d/kernel:5\1
/
_user_specified_nameAdam/v/conv3d_14/bias:5[1
/
_user_specified_nameAdam/m/conv3d_14/bias:7Z3
1
_user_specified_nameAdam/v/conv3d_14/kernel:7Y3
1
_user_specified_nameAdam/m/conv3d_14/kernel:5X1
/
_user_specified_nameAdam/v/conv3d_13/bias:5W1
/
_user_specified_nameAdam/m/conv3d_13/bias:7V3
1
_user_specified_nameAdam/v/conv3d_13/kernel:7U3
1
_user_specified_nameAdam/m/conv3d_13/kernel:4T0
.
_user_specified_nameAdam/v/conv3d_3/bias:4S0
.
_user_specified_nameAdam/m/conv3d_3/bias:6R2
0
_user_specified_nameAdam/v/conv3d_3/kernel:6Q2
0
_user_specified_nameAdam/m/conv3d_3/kernel:4P0
.
_user_specified_nameAdam/v/conv3d_2/bias:4O0
.
_user_specified_nameAdam/m/conv3d_2/bias:6N2
0
_user_specified_nameAdam/v/conv3d_2/kernel:6M2
0
_user_specified_nameAdam/m/conv3d_2/kernel:5L1
/
_user_specified_nameAdam/v/conv3d_12/bias:5K1
/
_user_specified_nameAdam/m/conv3d_12/bias:7J3
1
_user_specified_nameAdam/v/conv3d_12/kernel:7I3
1
_user_specified_nameAdam/m/conv3d_12/kernel:5H1
/
_user_specified_nameAdam/v/conv3d_11/bias:5G1
/
_user_specified_nameAdam/m/conv3d_11/bias:7F3
1
_user_specified_nameAdam/v/conv3d_11/kernel:7E3
1
_user_specified_nameAdam/m/conv3d_11/kernel:4D0
.
_user_specified_nameAdam/v/conv3d_5/bias:4C0
.
_user_specified_nameAdam/m/conv3d_5/bias:6B2
0
_user_specified_nameAdam/v/conv3d_5/kernel:6A2
0
_user_specified_nameAdam/m/conv3d_5/kernel:4@0
.
_user_specified_nameAdam/v/conv3d_4/bias:4?0
.
_user_specified_nameAdam/m/conv3d_4/bias:6>2
0
_user_specified_nameAdam/v/conv3d_4/kernel:6=2
0
_user_specified_nameAdam/m/conv3d_4/kernel:5<1
/
_user_specified_nameAdam/v/conv3d_10/bias:5;1
/
_user_specified_nameAdam/m/conv3d_10/bias:7:3
1
_user_specified_nameAdam/v/conv3d_10/kernel:793
1
_user_specified_nameAdam/m/conv3d_10/kernel:480
.
_user_specified_nameAdam/v/conv3d_9/bias:470
.
_user_specified_nameAdam/m/conv3d_9/bias:662
0
_user_specified_nameAdam/v/conv3d_9/kernel:652
0
_user_specified_nameAdam/m/conv3d_9/kernel:440
.
_user_specified_nameAdam/v/conv3d_8/bias:430
.
_user_specified_nameAdam/m/conv3d_8/bias:622
0
_user_specified_nameAdam/v/conv3d_8/kernel:612
0
_user_specified_nameAdam/m/conv3d_8/kernel:400
.
_user_specified_nameAdam/v/conv3d_7/bias:4/0
.
_user_specified_nameAdam/m/conv3d_7/bias:6.2
0
_user_specified_nameAdam/v/conv3d_7/kernel:6-2
0
_user_specified_nameAdam/m/conv3d_7/kernel:4,0
.
_user_specified_nameAdam/v/conv3d_6/bias:4+0
.
_user_specified_nameAdam/m/conv3d_6/bias:6*2
0
_user_specified_nameAdam/v/conv3d_6/kernel:6)2
0
_user_specified_nameAdam/m/conv3d_6/kernel:-()
'
_user_specified_namelearning_rate:)'%
#
_user_specified_name	iteration:.&*
(
_user_specified_nameconv3d_18/bias:0%,
*
_user_specified_nameconv3d_18/kernel:.$*
(
_user_specified_nameconv3d_17/bias:0#,
*
_user_specified_nameconv3d_17/kernel:."*
(
_user_specified_nameconv3d_16/bias:0!,
*
_user_specified_nameconv3d_16/kernel:. *
(
_user_specified_nameconv3d_15/bias:0,
*
_user_specified_nameconv3d_15/kernel:-)
'
_user_specified_nameconv3d_1/bias:/+
)
_user_specified_nameconv3d_1/kernel:+'
%
_user_specified_nameconv3d/bias:-)
'
_user_specified_nameconv3d/kernel:.*
(
_user_specified_nameconv3d_14/bias:0,
*
_user_specified_nameconv3d_14/kernel:.*
(
_user_specified_nameconv3d_13/bias:0,
*
_user_specified_nameconv3d_13/kernel:-)
'
_user_specified_nameconv3d_3/bias:/+
)
_user_specified_nameconv3d_3/kernel:-)
'
_user_specified_nameconv3d_2/bias:/+
)
_user_specified_nameconv3d_2/kernel:.*
(
_user_specified_nameconv3d_12/bias:0,
*
_user_specified_nameconv3d_12/kernel:.*
(
_user_specified_nameconv3d_11/bias:0,
*
_user_specified_nameconv3d_11/kernel:-)
'
_user_specified_nameconv3d_5/bias:/+
)
_user_specified_nameconv3d_5/kernel:-)
'
_user_specified_nameconv3d_4/bias:/+
)
_user_specified_nameconv3d_4/kernel:.
*
(
_user_specified_nameconv3d_10/bias:0	,
*
_user_specified_nameconv3d_10/kernel:-)
'
_user_specified_nameconv3d_9/bias:/+
)
_user_specified_nameconv3d_9/kernel:-)
'
_user_specified_nameconv3d_8/bias:/+
)
_user_specified_nameconv3d_8/kernel:-)
'
_user_specified_nameconv3d_7/bias:/+
)
_user_specified_nameconv3d_7/kernel:-)
'
_user_specified_nameconv3d_6/bias:/+
)
_user_specified_nameconv3d_6/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ж

D__inference_conv3d_12_layer_call_and_return_conditional_losses_16226

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ4m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ4S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџ4
 
_user_specified_nameinputs
е

C__inference_conv3d_3_layer_call_and_return_conditional_losses_16334

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџhm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџhS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџh
 
_user_specified_nameinputs
Љ
Ё
(__inference_conv3d_7_layer_call_fn_17418

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_7_layer_call_and_return_conditional_losses_16020{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџh<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџh: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name17414:%!

_user_specified_name17412:[ W
3
_output_shapes!
:џџџџџџџџџh
 
_user_specified_nameinputs
Ћ
Ђ
)__inference_conv3d_10_layer_call_fn_17518

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_10_layer_call_and_return_conditional_losses_16088{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name17514:%!

_user_specified_name17512:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Љ
Ё
(__inference_conv3d_4_layer_call_fn_17538

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_16072{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ4<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ4: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name17534:%!

_user_specified_name17532:[ W
3
_output_shapes!
:џџџџџџџџџ4
 
_user_specified_nameinputs
л
Z
"__inference__update_step_xla_17324
gradient&
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*+
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:T P
*
_output_shapes
:
"
_user_specified_name
gradient
ж

D__inference_conv3d_11_layer_call_and_return_conditional_losses_16194

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ4m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ4S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџ4
 
_user_specified_nameinputs
ж

D__inference_conv3d_10_layer_call_and_return_conditional_losses_16088

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
м
j
N__inference_average_pooling3d_5_layer_call_and_return_conditional_losses_17469

inputs
identityО
	AvgPool3D	AvgPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingVALID*
strides	

IdentityIdentityAvgPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
е

C__inference_conv3d_9_layer_call_and_return_conditional_losses_16056

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
м
j
N__inference_average_pooling3d_2_layer_call_and_return_conditional_losses_15985

inputs
identityО
	AvgPool3D	AvgPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingVALID*
strides	

IdentityIdentityAvgPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_17219
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
л

D__inference_conv3d_18_layer_call_and_return_conditional_losses_18162

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџаc
SigmoidSigmoidBiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџаg
IdentityIdentitySigmoid:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџаS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџа: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs

p
F__inference_concatenate_layer_call_and_return_conditional_losses_16182

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџ4c
IdentityIdentityconcat:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:џџџџџџџџџ4:џџџџџџџџџ4:[W
3
_output_shapes!
:џџџџџџџџџ4
 
_user_specified_nameinputs:[ W
3
_output_shapes!
:џџџџџџџџџ4
 
_user_specified_nameinputs

r
H__inference_concatenate_2_layer_call_and_return_conditional_losses_16562

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*4
_output_shapes"
 :џџџџџџџџџаd
IdentityIdentityconcat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџа"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџа:џџџџџџџџџа:\X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs:\ X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs
е

C__inference_conv3d_5_layer_call_and_return_conditional_losses_17639

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџ4\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ4m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ4S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџ4
 
_user_specified_nameinputs
Џ
Ђ
)__inference_conv3d_18_layer_call_fn_18151

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_18_layer_call_and_return_conditional_losses_16622|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџа<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџа: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name18147:%!

_user_specified_name18145:\ X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs
­
Ё
(__inference_conv3d_6_layer_call_fn_17388

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_6_layer_call_and_return_conditional_losses_16003|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџа<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџа: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name17384:%!

_user_specified_name17382:\ X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs
ж

D__inference_conv3d_14_layer_call_and_return_conditional_losses_16390

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџhm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџhS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџh
 
_user_specified_nameinputs
л
Z
"__inference__update_step_xla_17204
gradient&
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*+
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:T P
*
_output_shapes
:
"
_user_specified_name
gradient
л
Z
"__inference__update_step_xla_17294
gradient&
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*+
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:T P
*
_output_shapes
:
"
_user_specified_name
gradient
м
j
N__inference_average_pooling3d_2_layer_call_and_return_conditional_losses_17509

inputs
identityО
	AvgPool3D	AvgPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingVALID*
strides	

IdentityIdentityAvgPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
њ 
м

#__inference_signature_wrapper_17189
input_1%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:'
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:(

unknown_11:

unknown_12:(

unknown_13:

unknown_14:(

unknown_15:

unknown_16:(

unknown_17:

unknown_18:(

unknown_19:

unknown_20:(

unknown_21:

unknown_22:(

unknown_23:

unknown_24:(

unknown_25:

unknown_26:(

unknown_27:

unknown_28:(

unknown_29:

unknown_30:(

unknown_31:

unknown_32:(

unknown_33:

unknown_34:(

unknown_35:

unknown_36:
identityЂStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_15940|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџа<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:џџџџџџџџџа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%&!

_user_specified_name17185:%%!

_user_specified_name17183:%$!

_user_specified_name17181:%#!

_user_specified_name17179:%"!

_user_specified_name17177:%!!

_user_specified_name17175:% !

_user_specified_name17173:%!

_user_specified_name17171:%!

_user_specified_name17169:%!

_user_specified_name17167:%!

_user_specified_name17165:%!

_user_specified_name17163:%!

_user_specified_name17161:%!

_user_specified_name17159:%!

_user_specified_name17157:%!

_user_specified_name17155:%!

_user_specified_name17153:%!

_user_specified_name17151:%!

_user_specified_name17149:%!

_user_specified_name17147:%!

_user_specified_name17145:%!

_user_specified_name17143:%!

_user_specified_name17141:%!

_user_specified_name17139:%!

_user_specified_name17137:%!

_user_specified_name17135:%!

_user_specified_name17133:%!

_user_specified_name17131:%
!

_user_specified_name17129:%	!

_user_specified_name17127:%!

_user_specified_name17125:%!

_user_specified_name17123:%!

_user_specified_name17121:%!

_user_specified_name17119:%!

_user_specified_name17117:%!

_user_specified_name17115:%!

_user_specified_name17113:%!

_user_specified_name17111:] Y
4
_output_shapes"
 :џџџџџџџџџа
!
_user_specified_name	input_1
ЊN
f
J__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_16538

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :з
splitSplitsplit/split_dim:output:0inputs*
T0*
_output_shapesї
є:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Б
concatConcatV2split:output:0split:output:1split:output:2split:output:3split:output:4split:output:5split:output:6split:output:7split:output:8split:output:9split:output:10split:output:11concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџhS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ђ
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*Ш
_output_shapesЕ
В:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh:џџџџџџџџџh*
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :ѓ
concat_1ConcatV2split_1:output:0split_1:output:1split_1:output:2split_1:output:3split_1:output:4split_1:output:5split_1:output:6split_1:output:7split_1:output:8split_1:output:9split_1:output:10split_1:output:11split_1:output:12split_1:output:13concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџhS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*Ў
_output_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splithO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :й
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15split_2:output:16split_2:output:16split_2:output:17split_2:output:17split_2:output:18split_2:output:18split_2:output:19split_2:output:19split_2:output:20split_2:output:20split_2:output:21split_2:output:21split_2:output:22split_2:output:22split_2:output:23split_2:output:23split_2:output:24split_2:output:24split_2:output:25split_2:output:25split_2:output:26split_2:output:26split_2:output:27split_2:output:27split_2:output:28split_2:output:28split_2:output:29split_2:output:29split_2:output:30split_2:output:30split_2:output:31split_2:output:31split_2:output:32split_2:output:32split_2:output:33split_2:output:33split_2:output:34split_2:output:34split_2:output:35split_2:output:35split_2:output:36split_2:output:36split_2:output:37split_2:output:37split_2:output:38split_2:output:38split_2:output:39split_2:output:39split_2:output:40split_2:output:40split_2:output:41split_2:output:41split_2:output:42split_2:output:42split_2:output:43split_2:output:43split_2:output:44split_2:output:44split_2:output:45split_2:output:45split_2:output:46split_2:output:46split_2:output:47split_2:output:47split_2:output:48split_2:output:48split_2:output:49split_2:output:49split_2:output:50split_2:output:50split_2:output:51split_2:output:51split_2:output:52split_2:output:52split_2:output:53split_2:output:53split_2:output:54split_2:output:54split_2:output:55split_2:output:55split_2:output:56split_2:output:56split_2:output:57split_2:output:57split_2:output:58split_2:output:58split_2:output:59split_2:output:59split_2:output:60split_2:output:60split_2:output:61split_2:output:61split_2:output:62split_2:output:62split_2:output:63split_2:output:63split_2:output:64split_2:output:64split_2:output:65split_2:output:65split_2:output:66split_2:output:66split_2:output:67split_2:output:67split_2:output:68split_2:output:68split_2:output:69split_2:output:69split_2:output:70split_2:output:70split_2:output:71split_2:output:71split_2:output:72split_2:output:72split_2:output:73split_2:output:73split_2:output:74split_2:output:74split_2:output:75split_2:output:75split_2:output:76split_2:output:76split_2:output:77split_2:output:77split_2:output:78split_2:output:78split_2:output:79split_2:output:79split_2:output:80split_2:output:80split_2:output:81split_2:output:81split_2:output:82split_2:output:82split_2:output:83split_2:output:83split_2:output:84split_2:output:84split_2:output:85split_2:output:85split_2:output:86split_2:output:86split_2:output:87split_2:output:87split_2:output:88split_2:output:88split_2:output:89split_2:output:89split_2:output:90split_2:output:90split_2:output:91split_2:output:91split_2:output:92split_2:output:92split_2:output:93split_2:output:93split_2:output:94split_2:output:94split_2:output:95split_2:output:95split_2:output:96split_2:output:96split_2:output:97split_2:output:97split_2:output:98split_2:output:98split_2:output:99split_2:output:99split_2:output:100split_2:output:100split_2:output:101split_2:output:101split_2:output:102split_2:output:102split_2:output:103split_2:output:103concat_2/axis:output:0*
Nа*
T0*4
_output_shapes"
 :џџџџџџџџџаf
IdentityIdentityconcat_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџа"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџh:[ W
3
_output_shapes!
:џџџџџџџџџh
 
_user_specified_nameinputs
Љ

&__inference_conv3d_layer_call_fn_17890

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_16374|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџа<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџа: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name17886:%!

_user_specified_name17884:\ X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs
м
j
N__inference_average_pooling3d_4_layer_call_and_return_conditional_losses_17439

inputs
identityО
	AvgPool3D	AvgPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingVALID*
strides	

IdentityIdentityAvgPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ё
W
+__inference_concatenate_layer_call_fn_17645
inputs_0
inputs_1
identityЭ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_16182l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:џџџџџџџџџ4:џџџџџџџџџ4:]Y
3
_output_shapes!
:џџџџџџџџџ4
"
_user_specified_name
inputs_1:] Y
3
_output_shapes!
:џџџџџџџџџ4
"
_user_specified_name
inputs_0
!
о

%__inference_model_layer_call_fn_16901
input_1%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:'
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:(

unknown_11:

unknown_12:(

unknown_13:

unknown_14:(

unknown_15:

unknown_16:(

unknown_17:

unknown_18:(

unknown_19:

unknown_20:(

unknown_21:

unknown_22:(

unknown_23:

unknown_24:(

unknown_25:

unknown_26:(

unknown_27:

unknown_28:(

unknown_29:

unknown_30:(

unknown_31:

unknown_32:(

unknown_33:

unknown_34:(

unknown_35:

unknown_36:
identityЂStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_16739|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџа<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:џџџџџџџџџа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%&!

_user_specified_name16897:%%!

_user_specified_name16895:%$!

_user_specified_name16893:%#!

_user_specified_name16891:%"!

_user_specified_name16889:%!!

_user_specified_name16887:% !

_user_specified_name16885:%!

_user_specified_name16883:%!

_user_specified_name16881:%!

_user_specified_name16879:%!

_user_specified_name16877:%!

_user_specified_name16875:%!

_user_specified_name16873:%!

_user_specified_name16871:%!

_user_specified_name16869:%!

_user_specified_name16867:%!

_user_specified_name16865:%!

_user_specified_name16863:%!

_user_specified_name16861:%!

_user_specified_name16859:%!

_user_specified_name16857:%!

_user_specified_name16855:%!

_user_specified_name16853:%!

_user_specified_name16851:%!

_user_specified_name16849:%!

_user_specified_name16847:%!

_user_specified_name16845:%!

_user_specified_name16843:%
!

_user_specified_name16841:%	!

_user_specified_name16839:%!

_user_specified_name16837:%!

_user_specified_name16835:%!

_user_specified_name16833:%!

_user_specified_name16831:%!

_user_specified_name16829:%!

_user_specified_name16827:%!

_user_specified_name16825:%!

_user_specified_name16823:] Y
4
_output_shapes"
 :џџџџџџџџџа
!
_user_specified_name	input_1
Џ
Ђ
)__inference_conv3d_15_layer_call_fn_18091

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_15_layer_call_and_return_conditional_losses_16574|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџа<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџа: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name18087:%!

_user_specified_name18085:\ X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs
м

D__inference_conv3d_17_layer_call_and_return_conditional_losses_16606

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџаn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџаS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџа: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_17359
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
л

D__inference_conv3d_18_layer_call_and_return_conditional_losses_16622

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџаc
SigmoidSigmoidBiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџаg
IdentityIdentitySigmoid:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџаS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџа: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs
м

D__inference_conv3d_17_layer_call_and_return_conditional_losses_18142

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџаn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџаS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџа: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs
л
Z
"__inference__update_step_xla_17314
gradient&
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*+
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:T P
*
_output_shapes
:
"
_user_specified_name
gradient
о
K
/__inference_up_sampling3d_2_layer_call_fn_17906

inputs
identityХ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_16538m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџа"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџh:[ W
3
_output_shapes!
:џџџџџџџџџh
 
_user_specified_nameinputs
л

C__inference_conv3d_1_layer_call_and_return_conditional_losses_16550

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџаn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџаS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџа: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_17309
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
Ћ
Ђ
)__inference_conv3d_11_layer_call_fn_17661

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_11_layer_call_and_return_conditional_losses_16194{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ4<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ4: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name17657:%!

_user_specified_name17655:[ W
3
_output_shapes!
:џџџџџџџџџ4
 
_user_specified_nameinputs
Ћ
Ђ
)__inference_conv3d_13_layer_call_fn_17850

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_13_layer_call_and_return_conditional_losses_16358{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџh<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџh: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name17846:%!

_user_specified_name17844:[ W
3
_output_shapes!
:џџџџџџџџџh
 
_user_specified_nameinputs
л
Z
"__inference__update_step_xla_17254
gradient&
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*+
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:T P
*
_output_shapes
:
"
_user_specified_name
gradient
л

C__inference_conv3d_6_layer_call_and_return_conditional_losses_17399

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџаn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџаS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџа: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs
Љ
Ё
(__inference_conv3d_5_layer_call_fn_17628

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_16170{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ4<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ4: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name17624:%!

_user_specified_name17622:[ W
3
_output_shapes!
:џџџџџџџџџ4
 
_user_specified_nameinputs
м

D__inference_conv3d_15_layer_call_and_return_conditional_losses_18102

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџаn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџаS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџа: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs
ѕ
O
3__inference_average_pooling3d_3_layer_call_fn_17404

inputs
identityь
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling3d_3_layer_call_and_return_conditional_losses_15945
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
л
Z
"__inference__update_step_xla_17344
gradient&
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*+
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:T P
*
_output_shapes
:
"
_user_specified_name
gradient
Ћ
J
"__inference__update_step_xla_17209
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
ѕ
O
3__inference_average_pooling3d_4_layer_call_fn_17434

inputs
identityь
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling3d_4_layer_call_and_return_conditional_losses_15955
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
м

D__inference_conv3d_15_layer_call_and_return_conditional_losses_16574

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџаn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџаS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџа: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs

t
H__inference_concatenate_1_layer_call_and_return_conditional_losses_17841
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџhc
IdentityIdentityconcat:output:0*
T0*3
_output_shapes!
:џџџџџџџџџh"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:џџџџџџџџџh:џџџџџџџџџh:]Y
3
_output_shapes!
:џџџџџџџџџh
"
_user_specified_name
inputs_1:] Y
3
_output_shapes!
:џџџџџџџџџh
"
_user_specified_name
inputs_0
и
I
-__inference_up_sampling3d_layer_call_fn_17554

inputs
identityТ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_up_sampling3d_layer_call_and_return_conditional_losses_16158l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ћ
Ђ
)__inference_conv3d_12_layer_call_fn_17681

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_12_layer_call_and_return_conditional_losses_16226{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ4<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ4: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name17677:%!

_user_specified_name17675:[ W
3
_output_shapes!
:џџџџџџџџџ4
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_17229
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
ѕ
Y
-__inference_concatenate_1_layer_call_fn_17834
inputs_0
inputs_1
identityЯ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_16346l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџh"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:џџџџџџџџџh:џџџџџџџџџh:]Y
3
_output_shapes!
:џџџџџџџџџh
"
_user_specified_name
inputs_1:] Y
3
_output_shapes!
:џџџџџџџџџh
"
_user_specified_name
inputs_0
л
Z
"__inference__update_step_xla_17374
gradient&
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*+
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:T P
*
_output_shapes
:
"
_user_specified_name
gradient
л

C__inference_conv3d_6_layer_call_and_return_conditional_losses_16003

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџаn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџаS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџа: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs
л
Z
"__inference__update_step_xla_17234
gradient&
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*+
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:T P
*
_output_shapes
:
"
_user_specified_name
gradient
м
j
N__inference_average_pooling3d_1_layer_call_and_return_conditional_losses_17479

inputs
identityО
	AvgPool3D	AvgPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingVALID*
strides	

IdentityIdentityAvgPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
й
В
@__inference_model_layer_call_and_return_conditional_losses_16739
input_1,
conv3d_6_16632:
conv3d_6_16634:,
conv3d_7_16638:
conv3d_7_16640:,
conv3d_8_16644:
conv3d_8_16646:,
conv3d_9_16652:
conv3d_9_16654:,
conv3d_4_16657:
conv3d_4_16659:-
conv3d_10_16662:
conv3d_10_16664:,
conv3d_5_16668:
conv3d_5_16670:-
conv3d_11_16674:
conv3d_11_16676:,
conv3d_2_16679:
conv3d_2_16681:-
conv3d_12_16684:
conv3d_12_16686:,
conv3d_3_16690:
conv3d_3_16692:-
conv3d_13_16696:
conv3d_13_16698:*
conv3d_16701:
conv3d_16703:-
conv3d_14_16706:
conv3d_14_16708:,
conv3d_1_16712:
conv3d_1_16714:-
conv3d_15_16718:
conv3d_15_16720:-
conv3d_16_16723:
conv3d_16_16725:-
conv3d_17_16728:
conv3d_17_16730:-
conv3d_18_16733:
conv3d_18_16735:
identityЂconv3d/StatefulPartitionedCallЂ conv3d_1/StatefulPartitionedCallЂ!conv3d_10/StatefulPartitionedCallЂ!conv3d_11/StatefulPartitionedCallЂ!conv3d_12/StatefulPartitionedCallЂ!conv3d_13/StatefulPartitionedCallЂ!conv3d_14/StatefulPartitionedCallЂ!conv3d_15/StatefulPartitionedCallЂ!conv3d_16/StatefulPartitionedCallЂ!conv3d_17/StatefulPartitionedCallЂ!conv3d_18/StatefulPartitionedCallЂ conv3d_2/StatefulPartitionedCallЂ conv3d_3/StatefulPartitionedCallЂ conv3d_4/StatefulPartitionedCallЂ conv3d_5/StatefulPartitionedCallЂ conv3d_6/StatefulPartitionedCallЂ conv3d_7/StatefulPartitionedCallЂ conv3d_8/StatefulPartitionedCallЂ conv3d_9/StatefulPartitionedCallў
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_6_16632conv3d_6_16634*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_6_layer_call_and_return_conditional_losses_16003џ
#average_pooling3d_3/PartitionedCallPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling3d_3_layer_call_and_return_conditional_losses_15945Ђ
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling3d_3/PartitionedCall:output:0conv3d_7_16638conv3d_7_16640*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_7_layer_call_and_return_conditional_losses_16020џ
#average_pooling3d_4/PartitionedCallPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling3d_4_layer_call_and_return_conditional_losses_15955Ђ
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCall,average_pooling3d_4/PartitionedCall:output:0conv3d_8_16644conv3d_8_16646*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_8_layer_call_and_return_conditional_losses_16037н
#average_pooling3d_1/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling3d_1_layer_call_and_return_conditional_losses_15975џ
#average_pooling3d_5/PartitionedCallPartitionedCall)conv3d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling3d_5_layer_call_and_return_conditional_losses_15965
#average_pooling3d_2/PartitionedCallPartitionedCall,average_pooling3d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling3d_2_layer_call_and_return_conditional_losses_15985Ђ
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCall,average_pooling3d_5/PartitionedCall:output:0conv3d_9_16652conv3d_9_16654*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_9_layer_call_and_return_conditional_losses_16056Ђ
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall,average_pooling3d_2/PartitionedCall:output:0conv3d_4_16657conv3d_4_16659*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_16072Ѓ
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCall)conv3d_9/StatefulPartitionedCall:output:0conv3d_10_16662conv3d_10_16664*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_10_layer_call_and_return_conditional_losses_16088є
up_sampling3d/PartitionedCallPartitionedCall*conv3d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_up_sampling3d_layer_call_and_return_conditional_losses_16158
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0conv3d_5_16668conv3d_5_16670*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_16170
concatenate/PartitionedCallPartitionedCall&up_sampling3d/PartitionedCall:output:0)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_16182
!conv3d_11/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv3d_11_16674conv3d_11_16676*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_11_layer_call_and_return_conditional_losses_16194Ђ
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall,average_pooling3d_1/PartitionedCall:output:0conv3d_2_16679conv3d_2_16681*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_16210Є
!conv3d_12/StatefulPartitionedCallStatefulPartitionedCall*conv3d_11/StatefulPartitionedCall:output:0conv3d_12_16684conv3d_12_16686*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_12_layer_call_and_return_conditional_losses_16226ј
up_sampling3d_1/PartitionedCallPartitionedCall*conv3d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_16322
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0conv3d_3_16690conv3d_3_16692*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_16334
concatenate_1/PartitionedCallPartitionedCall(up_sampling3d_1/PartitionedCall:output:0)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_16346 
!conv3d_13/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv3d_13_16696conv3d_13_16698*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_13_layer_call_and_return_conditional_losses_16358і
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_16701conv3d_16703*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_16374Є
!conv3d_14/StatefulPartitionedCallStatefulPartitionedCall*conv3d_13/StatefulPartitionedCall:output:0conv3d_14_16706conv3d_14_16708*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_14_layer_call_and_return_conditional_losses_16390љ
up_sampling3d_2/PartitionedCallPartitionedCall*conv3d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_16538
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_16712conv3d_1_16714*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_16550
concatenate_2/PartitionedCallPartitionedCall(up_sampling3d_2/PartitionedCall:output:0)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_16562Ё
!conv3d_15/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv3d_15_16718conv3d_15_16720*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_15_layer_call_and_return_conditional_losses_16574Ѕ
!conv3d_16/StatefulPartitionedCallStatefulPartitionedCall*conv3d_15/StatefulPartitionedCall:output:0conv3d_16_16723conv3d_16_16725*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_16_layer_call_and_return_conditional_losses_16590Ѕ
!conv3d_17/StatefulPartitionedCallStatefulPartitionedCall*conv3d_16/StatefulPartitionedCall:output:0conv3d_17_16728conv3d_17_16730*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_17_layer_call_and_return_conditional_losses_16606Ѕ
!conv3d_18/StatefulPartitionedCallStatefulPartitionedCall*conv3d_17/StatefulPartitionedCall:output:0conv3d_18_16733conv3d_18_16735*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_18_layer_call_and_return_conditional_losses_16622
IdentityIdentity*conv3d_18/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџаТ
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall"^conv3d_11/StatefulPartitionedCall"^conv3d_12/StatefulPartitionedCall"^conv3d_13/StatefulPartitionedCall"^conv3d_14/StatefulPartitionedCall"^conv3d_15/StatefulPartitionedCall"^conv3d_16/StatefulPartitionedCall"^conv3d_17/StatefulPartitionedCall"^conv3d_18/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:џџџџџџџџџа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2F
!conv3d_10/StatefulPartitionedCall!conv3d_10/StatefulPartitionedCall2F
!conv3d_11/StatefulPartitionedCall!conv3d_11/StatefulPartitionedCall2F
!conv3d_12/StatefulPartitionedCall!conv3d_12/StatefulPartitionedCall2F
!conv3d_13/StatefulPartitionedCall!conv3d_13/StatefulPartitionedCall2F
!conv3d_14/StatefulPartitionedCall!conv3d_14/StatefulPartitionedCall2F
!conv3d_15/StatefulPartitionedCall!conv3d_15/StatefulPartitionedCall2F
!conv3d_16/StatefulPartitionedCall!conv3d_16/StatefulPartitionedCall2F
!conv3d_17/StatefulPartitionedCall!conv3d_17/StatefulPartitionedCall2F
!conv3d_18/StatefulPartitionedCall!conv3d_18/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2D
 conv3d_8/StatefulPartitionedCall conv3d_8/StatefulPartitionedCall2D
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall:%&!

_user_specified_name16735:%%!

_user_specified_name16733:%$!

_user_specified_name16730:%#!

_user_specified_name16728:%"!

_user_specified_name16725:%!!

_user_specified_name16723:% !

_user_specified_name16720:%!

_user_specified_name16718:%!

_user_specified_name16714:%!

_user_specified_name16712:%!

_user_specified_name16708:%!

_user_specified_name16706:%!

_user_specified_name16703:%!

_user_specified_name16701:%!

_user_specified_name16698:%!

_user_specified_name16696:%!

_user_specified_name16692:%!

_user_specified_name16690:%!

_user_specified_name16686:%!

_user_specified_name16684:%!

_user_specified_name16681:%!

_user_specified_name16679:%!

_user_specified_name16676:%!

_user_specified_name16674:%!

_user_specified_name16670:%!

_user_specified_name16668:%!

_user_specified_name16664:%!

_user_specified_name16662:%
!

_user_specified_name16659:%	!

_user_specified_name16657:%!

_user_specified_name16654:%!

_user_specified_name16652:%!

_user_specified_name16646:%!

_user_specified_name16644:%!

_user_specified_name16640:%!

_user_specified_name16638:%!

_user_specified_name16634:%!

_user_specified_name16632:] Y
4
_output_shapes"
 :џџџџџџџџџа
!
_user_specified_name	input_1
2
f
J__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_17808

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :з
splitSplitsplit/split_dim:output:0inputs*
T0*
_output_shapesї
є:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Б
concatConcatV2split:output:0split:output:1split:output:2split:output:3split:output:4split:output:5split:output:6split:output:7split:output:8split:output:9split:output:10split:output:11concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџ4S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ђ
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*Ш
_output_shapesЕ
В:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4:џџџџџџџџџ4*
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :ѓ
concat_1ConcatV2split_1:output:0split_1:output:1split_1:output:2split_1:output:3split_1:output:4split_1:output:5split_1:output:6split_1:output:7split_1:output:8split_1:output:9split_1:output:10split_1:output:11split_1:output:12split_1:output:13concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџ4S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :О
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*т
_output_shapesЯ
Ь:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split4O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15split_2:output:16split_2:output:16split_2:output:17split_2:output:17split_2:output:18split_2:output:18split_2:output:19split_2:output:19split_2:output:20split_2:output:20split_2:output:21split_2:output:21split_2:output:22split_2:output:22split_2:output:23split_2:output:23split_2:output:24split_2:output:24split_2:output:25split_2:output:25split_2:output:26split_2:output:26split_2:output:27split_2:output:27split_2:output:28split_2:output:28split_2:output:29split_2:output:29split_2:output:30split_2:output:30split_2:output:31split_2:output:31split_2:output:32split_2:output:32split_2:output:33split_2:output:33split_2:output:34split_2:output:34split_2:output:35split_2:output:35split_2:output:36split_2:output:36split_2:output:37split_2:output:37split_2:output:38split_2:output:38split_2:output:39split_2:output:39split_2:output:40split_2:output:40split_2:output:41split_2:output:41split_2:output:42split_2:output:42split_2:output:43split_2:output:43split_2:output:44split_2:output:44split_2:output:45split_2:output:45split_2:output:46split_2:output:46split_2:output:47split_2:output:47split_2:output:48split_2:output:48split_2:output:49split_2:output:49split_2:output:50split_2:output:50split_2:output:51split_2:output:51concat_2/axis:output:0*
Nh*
T0*3
_output_shapes!
:џџџџџџџџџhe
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:џџџџџџџџџh"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ4:[ W
3
_output_shapes!
:џџџџџџџџџ4
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_17259
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
Љ
Ё
(__inference_conv3d_8_layer_call_fn_17448

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ4*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv3d_8_layer_call_and_return_conditional_losses_16037{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџ4<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ4: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name17444:%!

_user_specified_name17442:[ W
3
_output_shapes!
:џџџџџџџџџ4
 
_user_specified_nameinputs
е

C__inference_conv3d_3_layer_call_and_return_conditional_losses_17828

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:џџџџџџџџџh\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:џџџџџџџџџhm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџhS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:џџџџџџџџџh
 
_user_specified_nameinputs
м
j
N__inference_average_pooling3d_3_layer_call_and_return_conditional_losses_15945

inputs
identityО
	AvgPool3D	AvgPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingVALID*
strides	

IdentityIdentityAvgPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_17369
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
ѕ
O
3__inference_average_pooling3d_2_layer_call_fn_17504

inputs
identityь
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling3d_2_layer_call_and_return_conditional_losses_15985
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_17379
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
Ћ
J
"__inference__update_step_xla_17319
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
л
Z
"__inference__update_step_xla_17224
gradient&
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*+
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:T P
*
_output_shapes
:
"
_user_specified_name
gradient
м

D__inference_conv3d_16_layer_call_and_return_conditional_losses_16590

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv3D/ReadVariableOp
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџа]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџаn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџаS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџа: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs
л
Z
"__inference__update_step_xla_17364
gradient&
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*+
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:T P
*
_output_shapes
:
"
_user_specified_name
gradient
Ћ
J
"__inference__update_step_xla_17269
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
$
d
H__inference_up_sampling3d_layer_call_and_return_conditional_losses_16158

inputs
identityQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :з
splitSplitsplit/split_dim:output:0inputs*
T0*
_output_shapesї
є:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Б
concatConcatV2split:output:0split:output:1split:output:2split:output:3split:output:4split:output:5split:output:6split:output:7split:output:8split:output:9split:output:10split:output:11concat/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ђ
split_1Splitsplit_1/split_dim:output:0concat:output:0*
T0*Ш
_output_shapesЕ
В:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :ѓ
concat_1ConcatV2split_1:output:0split_1:output:1split_1:output:2split_1:output:3split_1:output:4split_1:output:5split_1:output:6split_1:output:7split_1:output:8split_1:output:9split_1:output:10split_1:output:11split_1:output:12split_1:output:13concat_1/axis:output:0*
N*
T0*3
_output_shapes!
:џџџџџџџџџS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_2Splitsplit_2/split_dim:output:0concat_1:output:0*
T0*М
_output_shapesЉ
І:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :Л
concat_2ConcatV2split_2:output:0split_2:output:0split_2:output:1split_2:output:1split_2:output:2split_2:output:2split_2:output:3split_2:output:3split_2:output:4split_2:output:4split_2:output:5split_2:output:5split_2:output:6split_2:output:6split_2:output:7split_2:output:7split_2:output:8split_2:output:8split_2:output:9split_2:output:9split_2:output:10split_2:output:10split_2:output:11split_2:output:11split_2:output:12split_2:output:12split_2:output:13split_2:output:13split_2:output:14split_2:output:14split_2:output:15split_2:output:15split_2:output:16split_2:output:16split_2:output:17split_2:output:17split_2:output:18split_2:output:18split_2:output:19split_2:output:19split_2:output:20split_2:output:20split_2:output:21split_2:output:21split_2:output:22split_2:output:22split_2:output:23split_2:output:23split_2:output:24split_2:output:24split_2:output:25split_2:output:25concat_2/axis:output:0*
N4*
T0*3
_output_shapes!
:џџџџџџџџџ4e
IdentityIdentityconcat_2:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ:[ W
3
_output_shapes!
:џџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_17249
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
Ћ
J
"__inference__update_step_xla_17199
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
Ћ
Ђ
)__inference_conv3d_14_layer_call_fn_17870

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџh*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_14_layer_call_and_return_conditional_losses_16390{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:џџџџџџџџџh<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџh: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name17866:%!

_user_specified_name17864:[ W
3
_output_shapes!
:џџџџџџџџџh
 
_user_specified_nameinputs
Џ
Ђ
)__inference_conv3d_16_layer_call_fn_18111

inputs%
unknown:
	unknown_0:
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџа*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv3d_16_layer_call_and_return_conditional_losses_16590|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџа<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџа: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name18107:%!

_user_specified_name18105:\ X
4
_output_shapes"
 :џџџџџџџџџа
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_17279
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient"эL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ц
serving_defaultВ
H
input_1=
serving_default_input_1:0џџџџџџџџџаJ
	conv3d_18=
StatefulPartitionedCall:0џџџџџџџџџаtensorflow/serving/predict:м
	
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer_with_weights-8
layer-16
layer_with_weights-9
layer-17
layer-18
layer_with_weights-10
layer-19
layer-20
layer_with_weights-11
layer-21
layer_with_weights-12
layer-22
layer_with_weights-13
layer-23
layer-24
layer_with_weights-14
layer-25
layer-26
layer_with_weights-15
layer-27
layer_with_weights-16
layer-28
layer_with_weights-17
layer-29
layer_with_weights-18
layer-30
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses
&_default_save_signature
'	optimizer
(
signatures"
_tf_keras_network
"
_tf_keras_input_layer
н
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias
 1_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_layer
н
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias
 @_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_layer
н
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses

Mkernel
Nbias
 O_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
н
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

bkernel
cbias
 d_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_layer
н
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

qkernel
rbias
 s_jit_compiled_convolution_op"
_tf_keras_layer
н
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses

zkernel
{bias
 |_jit_compiled_convolution_op"
_tf_keras_layer
Ј
}	variables
~trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op"
_tf_keras_layer
ц
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses
Ёkernel
	Ђbias
!Ѓ_jit_compiled_convolution_op"
_tf_keras_layer
ц
Є	variables
Ѕtrainable_variables
Іregularization_losses
Ї	keras_api
Ј__call__
+Љ&call_and_return_all_conditional_losses
Њkernel
	Ћbias
!Ќ_jit_compiled_convolution_op"
_tf_keras_layer
Ћ
­	variables
Ўtrainable_variables
Џregularization_losses
А	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
З__call__
+И&call_and_return_all_conditional_losses
Йkernel
	Кbias
!Л_jit_compiled_convolution_op"
_tf_keras_layer
Ћ
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses
Шkernel
	Щbias
!Ъ_jit_compiled_convolution_op"
_tf_keras_layer
ц
Ы	variables
Ьtrainable_variables
Эregularization_losses
Ю	keras_api
Я__call__
+а&call_and_return_all_conditional_losses
бkernel
	вbias
!г_jit_compiled_convolution_op"
_tf_keras_layer
ц
д	variables
еtrainable_variables
жregularization_losses
з	keras_api
и__call__
+й&call_and_return_all_conditional_losses
кkernel
	лbias
!м_jit_compiled_convolution_op"
_tf_keras_layer
Ћ
н	variables
оtrainable_variables
пregularization_losses
р	keras_api
с__call__
+т&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
у	variables
фtrainable_variables
хregularization_losses
ц	keras_api
ч__call__
+ш&call_and_return_all_conditional_losses
щkernel
	ъbias
!ы_jit_compiled_convolution_op"
_tf_keras_layer
Ћ
ь	variables
эtrainable_variables
юregularization_losses
я	keras_api
№__call__
+ё&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
ђ	variables
ѓtrainable_variables
єregularization_losses
ѕ	keras_api
і__call__
+ї&call_and_return_all_conditional_losses
јkernel
	љbias
!њ_jit_compiled_convolution_op"
_tf_keras_layer
ц
ћ	variables
ќtrainable_variables
§regularization_losses
ў	keras_api
џ__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op"
_tf_keras_layer
ц
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op"
_tf_keras_layer
ц
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op"
_tf_keras_layer
р
/0
01
>2
?3
M4
N5
b6
c7
q8
r9
z10
{11
12
13
14
15
Ё16
Ђ17
Њ18
Ћ19
Й20
К21
Ш22
Щ23
б24
в25
к26
л27
щ28
ъ29
ј30
љ31
32
33
34
35
36
37"
trackable_list_wrapper
р
/0
01
>2
?3
M4
N5
b6
c7
q8
r9
z10
{11
12
13
14
15
Ё16
Ђ17
Њ18
Ћ19
Й20
К21
Ш22
Щ23
б24
в25
к26
л27
щ28
ъ29
ј30
љ31
32
33
34
35
36
37"
trackable_list_wrapper
 "
trackable_list_wrapper
Я
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
&_default_save_signature
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
С
trace_0
trace_12
%__inference_model_layer_call_fn_16820
%__inference_model_layer_call_fn_16901Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
ї
trace_0
trace_12М
@__inference_model_layer_call_and_return_conditional_losses_16629
@__inference_model_layer_call_and_return_conditional_losses_16739Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
ЫBШ
 __inference__wrapped_model_15940input_1"
В
FullArgSpec
args

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ѓ

_variables
 _iterations
Ё_learning_rate
Ђ_index_dict
Ѓ
_momentums
Є_velocities
Ѕ_update_step_xla"
experimentalOptimizer
-
Іserving_default"
signature_map
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Їnon_trainable_variables
Јlayers
Љmetrics
 Њlayer_regularization_losses
Ћlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
ф
Ќtrace_02Х
(__inference_conv3d_6_layer_call_fn_17388
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЌtrace_0
џ
­trace_02р
C__inference_conv3d_6_layer_call_and_return_conditional_losses_17399
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z­trace_0
-:+2conv3d_6/kernel
:2conv3d_6/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ўnon_trainable_variables
Џlayers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
я
Гtrace_02а
3__inference_average_pooling3d_3_layer_call_fn_17404
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zГtrace_0

Дtrace_02ы
N__inference_average_pooling3d_3_layer_call_and_return_conditional_losses_17409
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zДtrace_0
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
ф
Кtrace_02Х
(__inference_conv3d_7_layer_call_fn_17418
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zКtrace_0
џ
Лtrace_02р
C__inference_conv3d_7_layer_call_and_return_conditional_losses_17429
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЛtrace_0
-:+2conv3d_7/kernel
:2conv3d_7/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
я
Сtrace_02а
3__inference_average_pooling3d_4_layer_call_fn_17434
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zСtrace_0

Тtrace_02ы
N__inference_average_pooling3d_4_layer_call_and_return_conditional_losses_17439
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zТtrace_0
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
ф
Шtrace_02Х
(__inference_conv3d_8_layer_call_fn_17448
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zШtrace_0
џ
Щtrace_02р
C__inference_conv3d_8_layer_call_and_return_conditional_losses_17459
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЩtrace_0
-:+2conv3d_8/kernel
:2conv3d_8/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ъnon_trainable_variables
Ыlayers
Ьmetrics
 Эlayer_regularization_losses
Юlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
я
Яtrace_02а
3__inference_average_pooling3d_5_layer_call_fn_17464
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЯtrace_0

аtrace_02ы
N__inference_average_pooling3d_5_layer_call_and_return_conditional_losses_17469
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zаtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
я
жtrace_02а
3__inference_average_pooling3d_1_layer_call_fn_17474
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zжtrace_0

зtrace_02ы
N__inference_average_pooling3d_1_layer_call_and_return_conditional_losses_17479
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zзtrace_0
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
ф
нtrace_02Х
(__inference_conv3d_9_layer_call_fn_17488
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zнtrace_0
џ
оtrace_02р
C__inference_conv3d_9_layer_call_and_return_conditional_losses_17499
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zоtrace_0
-:+2conv3d_9/kernel
:2conv3d_9/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
пnon_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
я
фtrace_02а
3__inference_average_pooling3d_2_layer_call_fn_17504
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zфtrace_0

хtrace_02ы
N__inference_average_pooling3d_2_layer_call_and_return_conditional_losses_17509
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zхtrace_0
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
цnon_trainable_variables
чlayers
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
х
ыtrace_02Ц
)__inference_conv3d_10_layer_call_fn_17518
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zыtrace_0

ьtrace_02с
D__inference_conv3d_10_layer_call_and_return_conditional_losses_17529
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zьtrace_0
.:,2conv3d_10/kernel
:2conv3d_10/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
эnon_trainable_variables
юlayers
яmetrics
 №layer_regularization_losses
ёlayer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
ф
ђtrace_02Х
(__inference_conv3d_4_layer_call_fn_17538
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zђtrace_0
џ
ѓtrace_02р
C__inference_conv3d_4_layer_call_and_return_conditional_losses_17549
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zѓtrace_0
-:+2conv3d_4/kernel
:2conv3d_4/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
єnon_trainable_variables
ѕlayers
іmetrics
 їlayer_regularization_losses
јlayer_metrics
}	variables
~trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
щ
љtrace_02Ъ
-__inference_up_sampling3d_layer_call_fn_17554
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zљtrace_0

њtrace_02х
H__inference_up_sampling3d_layer_call_and_return_conditional_losses_17619
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zњtrace_0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ћnon_trainable_variables
ќlayers
§metrics
 ўlayer_regularization_losses
џlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ф
trace_02Х
(__inference_conv3d_5_layer_call_fn_17628
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
џ
trace_02р
C__inference_conv3d_5_layer_call_and_return_conditional_losses_17639
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
-:+2conv3d_5/kernel
:2conv3d_5/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ч
trace_02Ш
+__inference_concatenate_layer_call_fn_17645
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02у
F__inference_concatenate_layer_call_and_return_conditional_losses_17652
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
х
trace_02Ц
)__inference_conv3d_11_layer_call_fn_17661
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02с
D__inference_conv3d_11_layer_call_and_return_conditional_losses_17672
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
.:,2conv3d_11/kernel
:2conv3d_11/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
0
Ё0
Ђ1"
trackable_list_wrapper
0
Ё0
Ђ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
х
trace_02Ц
)__inference_conv3d_12_layer_call_fn_17681
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02с
D__inference_conv3d_12_layer_call_and_return_conditional_losses_17692
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
.:,2conv3d_12/kernel
:2conv3d_12/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
0
Њ0
Ћ1"
trackable_list_wrapper
0
Њ0
Ћ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Є	variables
Ѕtrainable_variables
Іregularization_losses
Ј__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
ф
trace_02Х
(__inference_conv3d_2_layer_call_fn_17701
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
џ
trace_02р
C__inference_conv3d_2_layer_call_and_return_conditional_losses_17712
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
-:+2conv3d_2/kernel
:2conv3d_2/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
­	variables
Ўtrainable_variables
Џregularization_losses
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
ы
Ѓtrace_02Ь
/__inference_up_sampling3d_1_layer_call_fn_17717
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЃtrace_0

Єtrace_02ч
J__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_17808
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЄtrace_0
0
Й0
К1"
trackable_list_wrapper
0
Й0
К1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
Г	variables
Дtrainable_variables
Еregularization_losses
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
ф
Њtrace_02Х
(__inference_conv3d_3_layer_call_fn_17817
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЊtrace_0
џ
Ћtrace_02р
C__inference_conv3d_3_layer_call_and_return_conditional_losses_17828
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЋtrace_0
-:+2conv3d_3/kernel
:2conv3d_3/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ќnon_trainable_variables
­layers
Ўmetrics
 Џlayer_regularization_losses
Аlayer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
щ
Бtrace_02Ъ
-__inference_concatenate_1_layer_call_fn_17834
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zБtrace_0

Вtrace_02х
H__inference_concatenate_1_layer_call_and_return_conditional_losses_17841
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zВtrace_0
0
Ш0
Щ1"
trackable_list_wrapper
0
Ш0
Щ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
Т	variables
Уtrainable_variables
Фregularization_losses
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
х
Иtrace_02Ц
)__inference_conv3d_13_layer_call_fn_17850
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zИtrace_0

Йtrace_02с
D__inference_conv3d_13_layer_call_and_return_conditional_losses_17861
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЙtrace_0
.:,2conv3d_13/kernel
:2conv3d_13/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
0
б0
в1"
trackable_list_wrapper
0
б0
в1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
Ы	variables
Ьtrainable_variables
Эregularization_losses
Я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
х
Пtrace_02Ц
)__inference_conv3d_14_layer_call_fn_17870
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zПtrace_0

Рtrace_02с
D__inference_conv3d_14_layer_call_and_return_conditional_losses_17881
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zРtrace_0
.:,2conv3d_14/kernel
:2conv3d_14/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
0
к0
л1"
trackable_list_wrapper
0
к0
л1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
д	variables
еtrainable_variables
жregularization_losses
и__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
т
Цtrace_02У
&__inference_conv3d_layer_call_fn_17890
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЦtrace_0
§
Чtrace_02о
A__inference_conv3d_layer_call_and_return_conditional_losses_17901
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЧtrace_0
+:)2conv3d/kernel
:2conv3d/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
н	variables
оtrainable_variables
пregularization_losses
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
ы
Эtrace_02Ь
/__inference_up_sampling3d_2_layer_call_fn_17906
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЭtrace_0

Юtrace_02ч
J__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_18049
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЮtrace_0
0
щ0
ъ1"
trackable_list_wrapper
0
щ0
ъ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
у	variables
фtrainable_variables
хregularization_losses
ч__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
ф
дtrace_02Х
(__inference_conv3d_1_layer_call_fn_18058
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zдtrace_0
џ
еtrace_02р
C__inference_conv3d_1_layer_call_and_return_conditional_losses_18069
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zеtrace_0
-:+2conv3d_1/kernel
:2conv3d_1/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
ь	variables
эtrainable_variables
юregularization_losses
№__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
щ
лtrace_02Ъ
-__inference_concatenate_2_layer_call_fn_18075
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zлtrace_0

мtrace_02х
H__inference_concatenate_2_layer_call_and_return_conditional_losses_18082
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zмtrace_0
0
ј0
љ1"
trackable_list_wrapper
0
ј0
љ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
нnon_trainable_variables
оlayers
пmetrics
 рlayer_regularization_losses
сlayer_metrics
ђ	variables
ѓtrainable_variables
єregularization_losses
і__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
х
тtrace_02Ц
)__inference_conv3d_15_layer_call_fn_18091
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zтtrace_0

уtrace_02с
D__inference_conv3d_15_layer_call_and_return_conditional_losses_18102
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zуtrace_0
.:,2conv3d_15/kernel
:2conv3d_15/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
фnon_trainable_variables
хlayers
цmetrics
 чlayer_regularization_losses
шlayer_metrics
ћ	variables
ќtrainable_variables
§regularization_losses
џ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
х
щtrace_02Ц
)__inference_conv3d_16_layer_call_fn_18111
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zщtrace_0

ъtrace_02с
D__inference_conv3d_16_layer_call_and_return_conditional_losses_18122
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zъtrace_0
.:,2conv3d_16/kernel
:2conv3d_16/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
х
№trace_02Ц
)__inference_conv3d_17_layer_call_fn_18131
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z№trace_0

ёtrace_02с
D__inference_conv3d_17_layer_call_and_return_conditional_losses_18142
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zёtrace_0
.:,2conv3d_17/kernel
:2conv3d_17/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ђnon_trainable_variables
ѓlayers
єmetrics
 ѕlayer_regularization_losses
іlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
х
їtrace_02Ц
)__inference_conv3d_18_layer_call_fn_18151
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zїtrace_0

јtrace_02с
D__inference_conv3d_18_layer_call_and_return_conditional_losses_18162
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zјtrace_0
.:,2conv3d_18/kernel
:2conv3d_18/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30"
trackable_list_wrapper
(
љ0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
фBс
%__inference_model_layer_call_fn_16820input_1"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
фBс
%__inference_model_layer_call_fn_16901input_1"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
@__inference_model_layer_call_and_return_conditional_losses_16629input_1"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
@__inference_model_layer_call_and_return_conditional_losses_16739input_1"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ы
 0
њ1
ћ2
ќ3
§4
ў5
џ6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
 39
Ё40
Ђ41
Ѓ42
Є43
Ѕ44
І45
Ї46
Ј47
Љ48
Њ49
Ћ50
Ќ51
­52
Ў53
Џ54
А55
Б56
В57
Г58
Д59
Е60
Ж61
З62
И63
Й64
К65
Л66
М67
Н68
О69
П70
Р71
С72
Т73
У74
Ф75
Х76"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
ь
њ0
ќ1
ў2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19
Ђ20
Є21
І22
Ј23
Њ24
Ќ25
Ў26
А27
В28
Д29
Ж30
И31
К32
М33
О34
Р35
Т36
Ф37"
trackable_list_wrapper
ь
ћ0
§1
џ2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
Ё19
Ѓ20
Ѕ21
Ї22
Љ23
Ћ24
­25
Џ26
Б27
Г28
Е29
З30
Й31
Л32
Н33
П34
С35
У36
Х37"
trackable_list_wrapper
э
Цtrace_0
Чtrace_1
Шtrace_2
Щtrace_3
Ъtrace_4
Ыtrace_5
Ьtrace_6
Эtrace_7
Юtrace_8
Яtrace_9
аtrace_10
бtrace_11
вtrace_12
гtrace_13
дtrace_14
еtrace_15
жtrace_16
зtrace_17
иtrace_18
йtrace_19
кtrace_20
лtrace_21
мtrace_22
нtrace_23
оtrace_24
пtrace_25
рtrace_26
сtrace_27
тtrace_28
уtrace_29
фtrace_30
хtrace_31
цtrace_32
чtrace_33
шtrace_34
щtrace_35
ъtrace_36
ыtrace_372
"__inference__update_step_xla_17194
"__inference__update_step_xla_17199
"__inference__update_step_xla_17204
"__inference__update_step_xla_17209
"__inference__update_step_xla_17214
"__inference__update_step_xla_17219
"__inference__update_step_xla_17224
"__inference__update_step_xla_17229
"__inference__update_step_xla_17234
"__inference__update_step_xla_17239
"__inference__update_step_xla_17244
"__inference__update_step_xla_17249
"__inference__update_step_xla_17254
"__inference__update_step_xla_17259
"__inference__update_step_xla_17264
"__inference__update_step_xla_17269
"__inference__update_step_xla_17274
"__inference__update_step_xla_17279
"__inference__update_step_xla_17284
"__inference__update_step_xla_17289
"__inference__update_step_xla_17294
"__inference__update_step_xla_17299
"__inference__update_step_xla_17304
"__inference__update_step_xla_17309
"__inference__update_step_xla_17314
"__inference__update_step_xla_17319
"__inference__update_step_xla_17324
"__inference__update_step_xla_17329
"__inference__update_step_xla_17334
"__inference__update_step_xla_17339
"__inference__update_step_xla_17344
"__inference__update_step_xla_17349
"__inference__update_step_xla_17354
"__inference__update_step_xla_17359
"__inference__update_step_xla_17364
"__inference__update_step_xla_17369
"__inference__update_step_xla_17374
"__inference__update_step_xla_17379Џ
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0zЦtrace_0zЧtrace_1zШtrace_2zЩtrace_3zЪtrace_4zЫtrace_5zЬtrace_6zЭtrace_7zЮtrace_8zЯtrace_9zаtrace_10zбtrace_11zвtrace_12zгtrace_13zдtrace_14zеtrace_15zжtrace_16zзtrace_17zиtrace_18zйtrace_19zкtrace_20zлtrace_21zмtrace_22zнtrace_23zоtrace_24zпtrace_25zрtrace_26zсtrace_27zтtrace_28zуtrace_29zфtrace_30zхtrace_31zцtrace_32zчtrace_33zшtrace_34zщtrace_35zъtrace_36zыtrace_37
ЯBЬ
#__inference_signature_wrapper_17189input_1"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs
	jinput_1
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBЯ
(__inference_conv3d_6_layer_call_fn_17388inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_conv3d_6_layer_call_and_return_conditional_losses_17399inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
нBк
3__inference_average_pooling3d_3_layer_call_fn_17404inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
N__inference_average_pooling3d_3_layer_call_and_return_conditional_losses_17409inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBЯ
(__inference_conv3d_7_layer_call_fn_17418inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_conv3d_7_layer_call_and_return_conditional_losses_17429inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
нBк
3__inference_average_pooling3d_4_layer_call_fn_17434inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
N__inference_average_pooling3d_4_layer_call_and_return_conditional_losses_17439inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBЯ
(__inference_conv3d_8_layer_call_fn_17448inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_conv3d_8_layer_call_and_return_conditional_losses_17459inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
нBк
3__inference_average_pooling3d_5_layer_call_fn_17464inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
N__inference_average_pooling3d_5_layer_call_and_return_conditional_losses_17469inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
нBк
3__inference_average_pooling3d_1_layer_call_fn_17474inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
N__inference_average_pooling3d_1_layer_call_and_return_conditional_losses_17479inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBЯ
(__inference_conv3d_9_layer_call_fn_17488inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_conv3d_9_layer_call_and_return_conditional_losses_17499inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
нBк
3__inference_average_pooling3d_2_layer_call_fn_17504inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
N__inference_average_pooling3d_2_layer_call_and_return_conditional_losses_17509inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
)__inference_conv3d_10_layer_call_fn_17518inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
D__inference_conv3d_10_layer_call_and_return_conditional_losses_17529inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBЯ
(__inference_conv3d_4_layer_call_fn_17538inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_conv3d_4_layer_call_and_return_conditional_losses_17549inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зBд
-__inference_up_sampling3d_layer_call_fn_17554inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђBя
H__inference_up_sampling3d_layer_call_and_return_conditional_losses_17619inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBЯ
(__inference_conv3d_5_layer_call_fn_17628inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_conv3d_5_layer_call_and_return_conditional_losses_17639inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBо
+__inference_concatenate_layer_call_fn_17645inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
F__inference_concatenate_layer_call_and_return_conditional_losses_17652inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
)__inference_conv3d_11_layer_call_fn_17661inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
D__inference_conv3d_11_layer_call_and_return_conditional_losses_17672inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
)__inference_conv3d_12_layer_call_fn_17681inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
D__inference_conv3d_12_layer_call_and_return_conditional_losses_17692inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBЯ
(__inference_conv3d_2_layer_call_fn_17701inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_conv3d_2_layer_call_and_return_conditional_losses_17712inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
йBж
/__inference_up_sampling3d_1_layer_call_fn_17717inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
єBё
J__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_17808inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBЯ
(__inference_conv3d_3_layer_call_fn_17817inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_conv3d_3_layer_call_and_return_conditional_losses_17828inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
уBр
-__inference_concatenate_1_layer_call_fn_17834inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_17841inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
)__inference_conv3d_13_layer_call_fn_17850inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
D__inference_conv3d_13_layer_call_and_return_conditional_losses_17861inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
)__inference_conv3d_14_layer_call_fn_17870inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
D__inference_conv3d_14_layer_call_and_return_conditional_losses_17881inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
аBЭ
&__inference_conv3d_layer_call_fn_17890inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ыBш
A__inference_conv3d_layer_call_and_return_conditional_losses_17901inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
йBж
/__inference_up_sampling3d_2_layer_call_fn_17906inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
єBё
J__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_18049inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBЯ
(__inference_conv3d_1_layer_call_fn_18058inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_conv3d_1_layer_call_and_return_conditional_losses_18069inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
уBр
-__inference_concatenate_2_layer_call_fn_18075inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_18082inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
)__inference_conv3d_15_layer_call_fn_18091inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
D__inference_conv3d_15_layer_call_and_return_conditional_losses_18102inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
)__inference_conv3d_16_layer_call_fn_18111inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
D__inference_conv3d_16_layer_call_and_return_conditional_losses_18122inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
)__inference_conv3d_17_layer_call_fn_18131inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
D__inference_conv3d_17_layer_call_and_return_conditional_losses_18142inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
)__inference_conv3d_18_layer_call_fn_18151inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
D__inference_conv3d_18_layer_call_and_return_conditional_losses_18162inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
ь	variables
э	keras_api

юtotal

яcount"
_tf_keras_metric
2:02Adam/m/conv3d_6/kernel
2:02Adam/v/conv3d_6/kernel
 :2Adam/m/conv3d_6/bias
 :2Adam/v/conv3d_6/bias
2:02Adam/m/conv3d_7/kernel
2:02Adam/v/conv3d_7/kernel
 :2Adam/m/conv3d_7/bias
 :2Adam/v/conv3d_7/bias
2:02Adam/m/conv3d_8/kernel
2:02Adam/v/conv3d_8/kernel
 :2Adam/m/conv3d_8/bias
 :2Adam/v/conv3d_8/bias
2:02Adam/m/conv3d_9/kernel
2:02Adam/v/conv3d_9/kernel
 :2Adam/m/conv3d_9/bias
 :2Adam/v/conv3d_9/bias
3:12Adam/m/conv3d_10/kernel
3:12Adam/v/conv3d_10/kernel
!:2Adam/m/conv3d_10/bias
!:2Adam/v/conv3d_10/bias
2:02Adam/m/conv3d_4/kernel
2:02Adam/v/conv3d_4/kernel
 :2Adam/m/conv3d_4/bias
 :2Adam/v/conv3d_4/bias
2:02Adam/m/conv3d_5/kernel
2:02Adam/v/conv3d_5/kernel
 :2Adam/m/conv3d_5/bias
 :2Adam/v/conv3d_5/bias
3:12Adam/m/conv3d_11/kernel
3:12Adam/v/conv3d_11/kernel
!:2Adam/m/conv3d_11/bias
!:2Adam/v/conv3d_11/bias
3:12Adam/m/conv3d_12/kernel
3:12Adam/v/conv3d_12/kernel
!:2Adam/m/conv3d_12/bias
!:2Adam/v/conv3d_12/bias
2:02Adam/m/conv3d_2/kernel
2:02Adam/v/conv3d_2/kernel
 :2Adam/m/conv3d_2/bias
 :2Adam/v/conv3d_2/bias
2:02Adam/m/conv3d_3/kernel
2:02Adam/v/conv3d_3/kernel
 :2Adam/m/conv3d_3/bias
 :2Adam/v/conv3d_3/bias
3:12Adam/m/conv3d_13/kernel
3:12Adam/v/conv3d_13/kernel
!:2Adam/m/conv3d_13/bias
!:2Adam/v/conv3d_13/bias
3:12Adam/m/conv3d_14/kernel
3:12Adam/v/conv3d_14/kernel
!:2Adam/m/conv3d_14/bias
!:2Adam/v/conv3d_14/bias
0:.2Adam/m/conv3d/kernel
0:.2Adam/v/conv3d/kernel
:2Adam/m/conv3d/bias
:2Adam/v/conv3d/bias
2:02Adam/m/conv3d_1/kernel
2:02Adam/v/conv3d_1/kernel
 :2Adam/m/conv3d_1/bias
 :2Adam/v/conv3d_1/bias
3:12Adam/m/conv3d_15/kernel
3:12Adam/v/conv3d_15/kernel
!:2Adam/m/conv3d_15/bias
!:2Adam/v/conv3d_15/bias
3:12Adam/m/conv3d_16/kernel
3:12Adam/v/conv3d_16/kernel
!:2Adam/m/conv3d_16/bias
!:2Adam/v/conv3d_16/bias
3:12Adam/m/conv3d_17/kernel
3:12Adam/v/conv3d_17/kernel
!:2Adam/m/conv3d_17/bias
!:2Adam/v/conv3d_17/bias
3:12Adam/m/conv3d_18/kernel
3:12Adam/v/conv3d_18/kernel
!:2Adam/m/conv3d_18/bias
!:2Adam/v/conv3d_18/bias
эBъ
"__inference__update_step_xla_17194gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17199gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17204gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17209gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17214gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17219gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17224gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17229gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17234gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17239gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17244gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17249gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17254gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17259gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17264gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17269gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17274gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17279gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17284gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17289gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17294gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17299gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17304gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17309gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17314gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17319gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17324gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17329gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17334gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17339gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17344gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17349gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17354gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17359gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17364gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17369gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17374gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
"__inference__update_step_xla_17379gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
ю0
я1"
trackable_list_wrapper
.
ь	variables"
_generic_user_object
:  (2total
:  (2countЎ
"__inference__update_step_xla_17194Ђ}
vЂs
%"
gradient
@=	)Ђ&
њ

p
` VariableSpec 
`р§ФСЧ

Њ "
 
"__inference__update_step_xla_17199f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
` ХСЧ

Њ "
 Ў
"__inference__update_step_xla_17204Ђ}
vЂs
%"
gradient
@=	)Ђ&
њ

p
` VariableSpec 
`РХСЧ

Њ "
 
"__inference__update_step_xla_17209f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`РХСЧ

Њ "
 Ў
"__inference__update_step_xla_17214Ђ}
vЂs
%"
gradient
@=	)Ђ&
њ

p
` VariableSpec 
`РўХСЧ

Њ "
 
"__inference__update_step_xla_17219f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
` §ХСЧ

Њ "
 Ў
"__inference__update_step_xla_17224Ђ}
vЂs
%"
gradient
@=	)Ђ&
њ

p
` VariableSpec 
`ЏЬСЧ

Њ "
 
"__inference__update_step_xla_17229f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`РБЬСЧ

Њ "
 Ў
"__inference__update_step_xla_17234Ђ}
vЂs
%"
gradient
@=	)Ђ&
њ

p
` VariableSpec 
` ёЬСЧ

Њ "
 
"__inference__update_step_xla_17239f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рѓЬСЧ

Њ "
 Ў
"__inference__update_step_xla_17244Ђ}
vЂs
%"
gradient
@=	)Ђ&
њ

p
` VariableSpec 
`ѕФСЧ

Њ "
 
"__inference__update_step_xla_17249f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`№ФСЧ

Њ "
 Ў
"__inference__update_step_xla_17254Ђ}
vЂs
%"
gradient
@=	)Ђ&
њ

p
` VariableSpec 
`РЯСЧ

Њ "
 
"__inference__update_step_xla_17259f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`РЪСЧ

Њ "
 Ў
"__inference__update_step_xla_17264Ђ}
vЂs
%"
gradient
@=	)Ђ&
њ

p
` VariableSpec 
` юЭСЧ

Њ "
 
"__inference__update_step_xla_17269f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`РєЭСЧ

Њ "
 Ў
"__inference__update_step_xla_17274Ђ}
vЂs
%"
gradient
@=	)Ђ&
њ

p
` VariableSpec 
`рЭСЧ

Њ "
 
"__inference__update_step_xla_17279f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
` ЭСЧ

Њ "
 Ў
"__inference__update_step_xla_17284Ђ}
vЂs
%"
gradient
@=	)Ђ&
њ

p
` VariableSpec 
`рзЗСЧ

Њ "
 
"__inference__update_step_xla_17289f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`ЯЗСЧ

Њ "
 Ў
"__inference__update_step_xla_17294Ђ}
vЂs
%"
gradient
@=	)Ђ&
њ

p
` VariableSpec 
`РФСЧ

Њ "
 
"__inference__update_step_xla_17299f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`РФСЧ

Њ "
 Ў
"__inference__update_step_xla_17304Ђ}
vЂs
%"
gradient
@=	)Ђ&
њ

p
` VariableSpec 
`рсеСЧ

Њ "
 
"__inference__update_step_xla_17309f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
` феСЧ

Њ "
 Ў
"__inference__update_step_xla_17314Ђ}
vЂs
%"
gradient
@=	)Ђ&
њ

p
` VariableSpec 
` ІтСЧ

Њ "
 
"__inference__update_step_xla_17319f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
` ееСЧ

Њ "
 Ў
"__inference__update_step_xla_17324Ђ}
vЂs
%"
gradient
@=	)Ђ&
њ

p
` VariableSpec 
`ркСЧ

Њ "
 
"__inference__update_step_xla_17329f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рФЄСЧ

Њ "
 Ў
"__inference__update_step_xla_17334Ђ}
vЂs
%"
gradient
@=	)Ђ&
њ

p
` VariableSpec 
`рЗСЧ

Њ "
 
"__inference__update_step_xla_17339f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`РЗСЧ

Њ "
 Ў
"__inference__update_step_xla_17344Ђ}
vЂs
%"
gradient
@=	)Ђ&
њ

p
` VariableSpec 
`руСЧ

Њ "
 
"__inference__update_step_xla_17349f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
` уСЧ

Њ "
 Ў
"__inference__update_step_xla_17354Ђ}
vЂs
%"
gradient
@=	)Ђ&
њ

p
` VariableSpec 
`РруСЧ

Њ "
 
"__inference__update_step_xla_17359f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`ууСЧ

Њ "
 Ў
"__inference__update_step_xla_17364Ђ}
vЂs
%"
gradient
@=	)Ђ&
њ

p
` VariableSpec 
`рптСЧ

Њ "
 
"__inference__update_step_xla_17369f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`РотСЧ

Њ "
 Ў
"__inference__update_step_xla_17374Ђ}
vЂs
%"
gradient
@=	)Ђ&
њ

p
` VariableSpec 
`ЊтСЧ

Њ "
 
"__inference__update_step_xla_17379f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
` КтСЧ

Њ "
 ъ
 __inference__wrapped_model_15940Х@/0>?MNbcz{qrЊЋЁЂЙКШЩклбвщъјљ=Ђ:
3Ђ0
.+
input_1џџџџџџџџџа
Њ "BЊ?
=
	conv3d_180-
	conv3d_18џџџџџџџџџа
N__inference_average_pooling3d_1_layer_call_and_return_conditional_losses_17479П_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "\ЂY
RO
tensor_0Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ь
3__inference_average_pooling3d_1_layer_call_fn_17474Д_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "QN
unknownAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
N__inference_average_pooling3d_2_layer_call_and_return_conditional_losses_17509П_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "\ЂY
RO
tensor_0Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ь
3__inference_average_pooling3d_2_layer_call_fn_17504Д_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "QN
unknownAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
N__inference_average_pooling3d_3_layer_call_and_return_conditional_losses_17409П_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "\ЂY
RO
tensor_0Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ь
3__inference_average_pooling3d_3_layer_call_fn_17404Д_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "QN
unknownAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
N__inference_average_pooling3d_4_layer_call_and_return_conditional_losses_17439П_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "\ЂY
RO
tensor_0Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ь
3__inference_average_pooling3d_4_layer_call_fn_17434Д_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "QN
unknownAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
N__inference_average_pooling3d_5_layer_call_and_return_conditional_losses_17469П_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "\ЂY
RO
tensor_0Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ь
3__inference_average_pooling3d_5_layer_call_fn_17464Д_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "QN
unknownAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџћ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_17841ЎrЂo
hЂe
c`
.+
inputs_0џџџџџџџџџh
.+
inputs_1џџџџџџџџџh
Њ "8Ђ5
.+
tensor_0џџџџџџџџџh
 е
-__inference_concatenate_1_layer_call_fn_17834ЃrЂo
hЂe
c`
.+
inputs_0џџџџџџџџџh
.+
inputs_1џџџџџџџџџh
Њ "-*
unknownџџџџџџџџџhў
H__inference_concatenate_2_layer_call_and_return_conditional_losses_18082БtЂq
jЂg
eb
/,
inputs_0џџџџџџџџџа
/,
inputs_1џџџџџџџџџа
Њ "9Ђ6
/,
tensor_0џџџџџџџџџа
 и
-__inference_concatenate_2_layer_call_fn_18075ІtЂq
jЂg
eb
/,
inputs_0џџџџџџџџџа
/,
inputs_1џџџџџџџџџа
Њ ".+
unknownџџџџџџџџџаљ
F__inference_concatenate_layer_call_and_return_conditional_losses_17652ЎrЂo
hЂe
c`
.+
inputs_0џџџџџџџџџ4
.+
inputs_1џџџџџџџџџ4
Њ "8Ђ5
.+
tensor_0џџџџџџџџџ4
 г
+__inference_concatenate_layer_call_fn_17645ЃrЂo
hЂe
c`
.+
inputs_0џџџџџџџџџ4
.+
inputs_1џџџџџџџџџ4
Њ "-*
unknownџџџџџџџџџ4У
D__inference_conv3d_10_layer_call_and_return_conditional_losses_17529{qr;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "8Ђ5
.+
tensor_0џџџџџџџџџ
 
)__inference_conv3d_10_layer_call_fn_17518pqr;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "-*
unknownџџџџџџџџџХ
D__inference_conv3d_11_layer_call_and_return_conditional_losses_17672};Ђ8
1Ђ.
,)
inputsџџџџџџџџџ4
Њ "8Ђ5
.+
tensor_0џџџџџџџџџ4
 
)__inference_conv3d_11_layer_call_fn_17661r;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ4
Њ "-*
unknownџџџџџџџџџ4Х
D__inference_conv3d_12_layer_call_and_return_conditional_losses_17692}ЁЂ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ4
Њ "8Ђ5
.+
tensor_0џџџџџџџџџ4
 
)__inference_conv3d_12_layer_call_fn_17681rЁЂ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ4
Њ "-*
unknownџџџџџџџџџ4Х
D__inference_conv3d_13_layer_call_and_return_conditional_losses_17861}ШЩ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџh
Њ "8Ђ5
.+
tensor_0џџџџџџџџџh
 
)__inference_conv3d_13_layer_call_fn_17850rШЩ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџh
Њ "-*
unknownџџџџџџџџџhХ
D__inference_conv3d_14_layer_call_and_return_conditional_losses_17881}бв;Ђ8
1Ђ.
,)
inputsџџџџџџџџџh
Њ "8Ђ5
.+
tensor_0џџџџџџџџџh
 
)__inference_conv3d_14_layer_call_fn_17870rбв;Ђ8
1Ђ.
,)
inputsџџџџџџџџџh
Њ "-*
unknownџџџџџџџџџhЧ
D__inference_conv3d_15_layer_call_and_return_conditional_losses_18102јљ<Ђ9
2Ђ/
-*
inputsџџџџџџџџџа
Њ "9Ђ6
/,
tensor_0џџџџџџџџџа
 Ё
)__inference_conv3d_15_layer_call_fn_18091tјљ<Ђ9
2Ђ/
-*
inputsџџџџџџџџџа
Њ ".+
unknownџџџџџџџџџаЧ
D__inference_conv3d_16_layer_call_and_return_conditional_losses_18122<Ђ9
2Ђ/
-*
inputsџџџџџџџџџа
Њ "9Ђ6
/,
tensor_0џџџџџџџџџа
 Ё
)__inference_conv3d_16_layer_call_fn_18111t<Ђ9
2Ђ/
-*
inputsџџџџџџџџџа
Њ ".+
unknownџџџџџџџџџаЧ
D__inference_conv3d_17_layer_call_and_return_conditional_losses_18142<Ђ9
2Ђ/
-*
inputsџџџџџџџџџа
Њ "9Ђ6
/,
tensor_0џџџџџџџџџа
 Ё
)__inference_conv3d_17_layer_call_fn_18131t<Ђ9
2Ђ/
-*
inputsџџџџџџџџџа
Њ ".+
unknownџџџџџџџџџаЧ
D__inference_conv3d_18_layer_call_and_return_conditional_losses_18162<Ђ9
2Ђ/
-*
inputsџџџџџџџџџа
Њ "9Ђ6
/,
tensor_0џџџџџџџџџа
 Ё
)__inference_conv3d_18_layer_call_fn_18151t<Ђ9
2Ђ/
-*
inputsџџџџџџџџџа
Њ ".+
unknownџџџџџџџџџаЦ
C__inference_conv3d_1_layer_call_and_return_conditional_losses_18069щъ<Ђ9
2Ђ/
-*
inputsџџџџџџџџџа
Њ "9Ђ6
/,
tensor_0џџџџџџџџџа
  
(__inference_conv3d_1_layer_call_fn_18058tщъ<Ђ9
2Ђ/
-*
inputsџџџџџџџџџа
Њ ".+
unknownџџџџџџџџџаФ
C__inference_conv3d_2_layer_call_and_return_conditional_losses_17712}ЊЋ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџh
Њ "8Ђ5
.+
tensor_0џџџџџџџџџh
 
(__inference_conv3d_2_layer_call_fn_17701rЊЋ;Ђ8
1Ђ.
,)
inputsџџџџџџџџџh
Њ "-*
unknownџџџџџџџџџhФ
C__inference_conv3d_3_layer_call_and_return_conditional_losses_17828}ЙК;Ђ8
1Ђ.
,)
inputsџџџџџџџџџh
Њ "8Ђ5
.+
tensor_0џџџџџџџџџh
 
(__inference_conv3d_3_layer_call_fn_17817rЙК;Ђ8
1Ђ.
,)
inputsџџџџџџџџџh
Њ "-*
unknownџџџџџџџџџhТ
C__inference_conv3d_4_layer_call_and_return_conditional_losses_17549{z{;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ4
Њ "8Ђ5
.+
tensor_0џџџџџџџџџ4
 
(__inference_conv3d_4_layer_call_fn_17538pz{;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ4
Њ "-*
unknownџџџџџџџџџ4Ф
C__inference_conv3d_5_layer_call_and_return_conditional_losses_17639};Ђ8
1Ђ.
,)
inputsџџџџџџџџџ4
Њ "8Ђ5
.+
tensor_0џџџџџџџџџ4
 
(__inference_conv3d_5_layer_call_fn_17628r;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ4
Њ "-*
unknownџџџџџџџџџ4Ф
C__inference_conv3d_6_layer_call_and_return_conditional_losses_17399}/0<Ђ9
2Ђ/
-*
inputsџџџџџџџџџа
Њ "9Ђ6
/,
tensor_0џџџџџџџџџа
 
(__inference_conv3d_6_layer_call_fn_17388r/0<Ђ9
2Ђ/
-*
inputsџџџџџџџџџа
Њ ".+
unknownџџџџџџџџџаТ
C__inference_conv3d_7_layer_call_and_return_conditional_losses_17429{>?;Ђ8
1Ђ.
,)
inputsџџџџџџџџџh
Њ "8Ђ5
.+
tensor_0џџџџџџџџџh
 
(__inference_conv3d_7_layer_call_fn_17418p>?;Ђ8
1Ђ.
,)
inputsџџџџџџџџџh
Њ "-*
unknownџџџџџџџџџhТ
C__inference_conv3d_8_layer_call_and_return_conditional_losses_17459{MN;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ4
Њ "8Ђ5
.+
tensor_0џџџџџџџџџ4
 
(__inference_conv3d_8_layer_call_fn_17448pMN;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ4
Њ "-*
unknownџџџџџџџџџ4Т
C__inference_conv3d_9_layer_call_and_return_conditional_losses_17499{bc;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "8Ђ5
.+
tensor_0џџџџџџџџџ
 
(__inference_conv3d_9_layer_call_fn_17488pbc;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "-*
unknownџџџџџџџџџФ
A__inference_conv3d_layer_call_and_return_conditional_losses_17901кл<Ђ9
2Ђ/
-*
inputsџџџџџџџџџа
Њ "9Ђ6
/,
tensor_0џџџџџџџџџа
 
&__inference_conv3d_layer_call_fn_17890tкл<Ђ9
2Ђ/
-*
inputsџџџџџџџџџа
Њ ".+
unknownџџџџџџџџџа
@__inference_model_layer_call_and_return_conditional_losses_16629Ф@/0>?MNbcz{qrЊЋЁЂЙКШЩклбвщъјљEЂB
;Ђ8
.+
input_1џџџџџџџџџа
p

 
Њ "9Ђ6
/,
tensor_0џџџџџџџџџа
 
@__inference_model_layer_call_and_return_conditional_losses_16739Ф@/0>?MNbcz{qrЊЋЁЂЙКШЩклбвщъјљEЂB
;Ђ8
.+
input_1џџџџџџџџџа
p 

 
Њ "9Ђ6
/,
tensor_0џџџџџџџџџа
 у
%__inference_model_layer_call_fn_16820Й@/0>?MNbcz{qrЊЋЁЂЙКШЩклбвщъјљEЂB
;Ђ8
.+
input_1џџџџџџџџџа
p

 
Њ ".+
unknownџџџџџџџџџау
%__inference_model_layer_call_fn_16901Й@/0>?MNbcz{qrЊЋЁЂЙКШЩклбвщъјљEЂB
;Ђ8
.+
input_1џџџџџџџџџа
p 

 
Њ ".+
unknownџџџџџџџџџај
#__inference_signature_wrapper_17189а@/0>?MNbcz{qrЊЋЁЂЙКШЩклбвщъјљHЂE
Ђ 
>Њ;
9
input_1.+
input_1џџџџџџџџџа"BЊ?
=
	conv3d_180-
	conv3d_18џџџџџџџџџаХ
J__inference_up_sampling3d_1_layer_call_and_return_conditional_losses_17808w;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ4
Њ "8Ђ5
.+
tensor_0џџџџџџџџџh
 
/__inference_up_sampling3d_1_layer_call_fn_17717l;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ4
Њ "-*
unknownџџџџџџџџџhЦ
J__inference_up_sampling3d_2_layer_call_and_return_conditional_losses_18049x;Ђ8
1Ђ.
,)
inputsџџџџџџџџџh
Њ "9Ђ6
/,
tensor_0џџџџџџџџџа
  
/__inference_up_sampling3d_2_layer_call_fn_17906m;Ђ8
1Ђ.
,)
inputsџџџџџџџџџh
Њ ".+
unknownџџџџџџџџџаУ
H__inference_up_sampling3d_layer_call_and_return_conditional_losses_17619w;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "8Ђ5
.+
tensor_0џџџџџџџџџ4
 
-__inference_up_sampling3d_layer_call_fn_17554l;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ
Њ "-*
unknownџџџџџџџџџ4