
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring ��
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628�
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
�
Adam/v/output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/v/output_layer/bias
�
,Adam/v/output_layer/bias/Read/ReadVariableOpReadVariableOpAdam/v/output_layer/bias*
_output_shapes
:*
dtype0
�
Adam/m/output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/m/output_layer/bias
�
,Adam/m/output_layer/bias/Read/ReadVariableOpReadVariableOpAdam/m/output_layer/bias*
_output_shapes
:*
dtype0
�
Adam/v/output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameAdam/v/output_layer/kernel
�
.Adam/v/output_layer/kernel/Read/ReadVariableOpReadVariableOpAdam/v/output_layer/kernel*
_output_shapes

:*
dtype0
�
Adam/m/output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameAdam/m/output_layer/kernel
�
.Adam/m/output_layer/kernel/Read/ReadVariableOpReadVariableOpAdam/m/output_layer/kernel*
_output_shapes

:*
dtype0
t
Adam/v/sd/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/v/sd/bias
m
"Adam/v/sd/bias/Read/ReadVariableOpReadVariableOpAdam/v/sd/bias*
_output_shapes
:*
dtype0
t
Adam/m/sd/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/m/sd/bias
m
"Adam/m/sd/bias/Read/ReadVariableOpReadVariableOpAdam/m/sd/bias*
_output_shapes
:*
dtype0
|
Adam/v/sd/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_nameAdam/v/sd/kernel
u
$Adam/v/sd/kernel/Read/ReadVariableOpReadVariableOpAdam/v/sd/kernel*
_output_shapes

:*
dtype0
|
Adam/m/sd/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_nameAdam/m/sd/kernel
u
$Adam/m/sd/kernel/Read/ReadVariableOpReadVariableOpAdam/m/sd/kernel*
_output_shapes

:*
dtype0
�
Adam/v/outputlayer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/v/outputlayer/bias

+Adam/v/outputlayer/bias/Read/ReadVariableOpReadVariableOpAdam/v/outputlayer/bias*
_output_shapes
:*
dtype0
�
Adam/m/outputlayer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/m/outputlayer/bias

+Adam/m/outputlayer/bias/Read/ReadVariableOpReadVariableOpAdam/m/outputlayer/bias*
_output_shapes
:*
dtype0
�
Adam/v/outputlayer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameAdam/v/outputlayer/kernel
�
-Adam/v/outputlayer/kernel/Read/ReadVariableOpReadVariableOpAdam/v/outputlayer/kernel*
_output_shapes

:*
dtype0
�
Adam/m/outputlayer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameAdam/m/outputlayer/kernel
�
-Adam/m/outputlayer/kernel/Read/ReadVariableOpReadVariableOpAdam/m/outputlayer/kernel*
_output_shapes

:*
dtype0
|
Adam/v/output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/v/output/bias
u
&Adam/v/output/bias/Read/ReadVariableOpReadVariableOpAdam/v/output/bias*
_output_shapes
:*
dtype0
|
Adam/m/output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/m/output/bias
u
&Adam/m/output/bias/Read/ReadVariableOpReadVariableOpAdam/m/output/bias*
_output_shapes
:*
dtype0
�
Adam/v/output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/v/output/kernel
}
(Adam/v/output/kernel/Read/ReadVariableOpReadVariableOpAdam/v/output/kernel*
_output_shapes

:*
dtype0
�
Adam/m/output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/m/output/kernel
}
(Adam/m/output/kernel/Read/ReadVariableOpReadVariableOpAdam/m/output/kernel*
_output_shapes

:*
dtype0
|
Adam/v/layer4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/v/layer4/bias
u
&Adam/v/layer4/bias/Read/ReadVariableOpReadVariableOpAdam/v/layer4/bias*
_output_shapes
:*
dtype0
|
Adam/m/layer4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/m/layer4/bias
u
&Adam/m/layer4/bias/Read/ReadVariableOpReadVariableOpAdam/m/layer4/bias*
_output_shapes
:*
dtype0
�
Adam/v/layer4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_nameAdam/v/layer4/kernel
}
(Adam/v/layer4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/layer4/kernel*
_output_shapes

: *
dtype0
�
Adam/m/layer4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_nameAdam/m/layer4/kernel
}
(Adam/m/layer4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/layer4/kernel*
_output_shapes

: *
dtype0
|
Adam/v/layer3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/v/layer3/bias
u
&Adam/v/layer3/bias/Read/ReadVariableOpReadVariableOpAdam/v/layer3/bias*
_output_shapes
: *
dtype0
|
Adam/m/layer3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/m/layer3/bias
u
&Adam/m/layer3/bias/Read/ReadVariableOpReadVariableOpAdam/m/layer3/bias*
_output_shapes
: *
dtype0
�
Adam/v/layer3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *%
shared_nameAdam/v/layer3/kernel
}
(Adam/v/layer3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/layer3/kernel*
_output_shapes

:@ *
dtype0
�
Adam/m/layer3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *%
shared_nameAdam/m/layer3/kernel
}
(Adam/m/layer3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/layer3/kernel*
_output_shapes

:@ *
dtype0
|
Adam/v/layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/v/layer2/bias
u
&Adam/v/layer2/bias/Read/ReadVariableOpReadVariableOpAdam/v/layer2/bias*
_output_shapes
:@*
dtype0
|
Adam/m/layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/m/layer2/bias
u
&Adam/m/layer2/bias/Read/ReadVariableOpReadVariableOpAdam/m/layer2/bias*
_output_shapes
:@*
dtype0
�
Adam/v/layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*%
shared_nameAdam/v/layer2/kernel
~
(Adam/v/layer2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/layer2/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/m/layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*%
shared_nameAdam/m/layer2/kernel
~
(Adam/m/layer2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/layer2/kernel*
_output_shapes
:	�@*
dtype0
}
Adam/v/layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/v/layer1/bias
v
&Adam/v/layer1/bias/Read/ReadVariableOpReadVariableOpAdam/v/layer1/bias*
_output_shapes	
:�*
dtype0
}
Adam/m/layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/m/layer1/bias
v
&Adam/m/layer1/bias/Read/ReadVariableOpReadVariableOpAdam/m/layer1/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*%
shared_nameAdam/v/layer1/kernel
~
(Adam/v/layer1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/layer1/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*%
shared_nameAdam/m/layer1/kernel
~
(Adam/m/layer1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/layer1/kernel*
_output_shapes
:	�*
dtype0
�
Adam/v/inputlayer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/inputlayer/bias
}
*Adam/v/inputlayer/bias/Read/ReadVariableOpReadVariableOpAdam/v/inputlayer/bias*
_output_shapes
:*
dtype0
�
Adam/m/inputlayer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/inputlayer/bias
}
*Adam/m/inputlayer/bias/Read/ReadVariableOpReadVariableOpAdam/m/inputlayer/bias*
_output_shapes
:*
dtype0
�
Adam/v/inputlayer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/v/inputlayer/kernel
�
,Adam/v/inputlayer/kernel/Read/ReadVariableOpReadVariableOpAdam/v/inputlayer/kernel*
_output_shapes

:*
dtype0
�
Adam/m/inputlayer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/m/inputlayer/kernel
�
,Adam/m/inputlayer/kernel/Read/ReadVariableOpReadVariableOpAdam/m/inputlayer/kernel*
_output_shapes

:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
z
output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameoutput_layer/bias
s
%output_layer/bias/Read/ReadVariableOpReadVariableOpoutput_layer/bias*
_output_shapes
:*
dtype0
�
output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameoutput_layer/kernel
{
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*
_output_shapes

:*
dtype0
f
sd/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	sd/bias
_
sd/bias/Read/ReadVariableOpReadVariableOpsd/bias*
_output_shapes
:*
dtype0
n
	sd/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name	sd/kernel
g
sd/kernel/Read/ReadVariableOpReadVariableOp	sd/kernel*
_output_shapes

:*
dtype0
x
outputlayer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameoutputlayer/bias
q
$outputlayer/bias/Read/ReadVariableOpReadVariableOpoutputlayer/bias*
_output_shapes
:*
dtype0
�
outputlayer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*#
shared_nameoutputlayer/kernel
y
&outputlayer/kernel/Read/ReadVariableOpReadVariableOpoutputlayer/kernel*
_output_shapes

:*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:*
dtype0
n
layer4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer4/bias
g
layer4/bias/Read/ReadVariableOpReadVariableOplayer4/bias*
_output_shapes
:*
dtype0
v
layer4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namelayer4/kernel
o
!layer4/kernel/Read/ReadVariableOpReadVariableOplayer4/kernel*
_output_shapes

: *
dtype0
n
layer3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelayer3/bias
g
layer3/bias/Read/ReadVariableOpReadVariableOplayer3/bias*
_output_shapes
: *
dtype0
v
layer3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namelayer3/kernel
o
!layer3/kernel/Read/ReadVariableOpReadVariableOplayer3/kernel*
_output_shapes

:@ *
dtype0
n
layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namelayer2/bias
g
layer2/bias/Read/ReadVariableOpReadVariableOplayer2/bias*
_output_shapes
:@*
dtype0
w
layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_namelayer2/kernel
p
!layer2/kernel/Read/ReadVariableOpReadVariableOplayer2/kernel*
_output_shapes
:	�@*
dtype0
o
layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namelayer1/bias
h
layer1/bias/Read/ReadVariableOpReadVariableOplayer1/bias*
_output_shapes	
:�*
dtype0
w
layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namelayer1/kernel
p
!layer1/kernel/Read/ReadVariableOpReadVariableOplayer1/kernel*
_output_shapes
:	�*
dtype0
v
inputlayer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameinputlayer/bias
o
#inputlayer/bias/Read/ReadVariableOpReadVariableOpinputlayer/bias*
_output_shapes
:*
dtype0
~
inputlayer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameinputlayer/kernel
w
%inputlayer/kernel/Read/ReadVariableOpReadVariableOpinputlayer/kernel*
_output_shapes

:*
dtype0
�
 serving_default_inputlayer_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCall serving_default_inputlayer_inputinputlayer/kernelinputlayer/biaslayer1/kernellayer1/biaslayer2/kernellayer2/biaslayer3/kernellayer3/biaslayer4/kernellayer4/biasoutput/kerneloutput/biasoutputlayer/kerneloutputlayer/bias	sd/kernelsd/biasoutput_layer/kerneloutput_layer/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_202481

NoOpNoOp
�d
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�d
value�dB�d B�d
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
	layer_with_weights-8
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias*
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias*
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses

1kernel
2bias*
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias*
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses

Akernel
Bbias*
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

Ikernel
Jbias*
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias*
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias*
�
0
1
!2
"3
)4
*5
16
27
98
:9
A10
B11
I12
J13
Q14
R15
Y16
Z17*
�
0
1
!2
"3
)4
*5
16
27
98
:9
A10
B11
I12
J13
Q14
R15
Y16
Z17*
* 
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

`trace_0
atrace_1* 

btrace_0
ctrace_1* 
* 
�
d
_variables
e_iterations
f_learning_rate
g_index_dict
h
_momentums
i_velocities
j_update_step_xla*

kserving_default* 

0
1*

0
1*
* 
�
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

qtrace_0* 

rtrace_0* 
a[
VARIABLE_VALUEinputlayer/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEinputlayer/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

!0
"1*

!0
"1*
* 
�
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*

xtrace_0* 

ytrace_0* 
]W
VARIABLE_VALUElayer1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

)0
*1*

)0
*1*
* 
�
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

trace_0* 

�trace_0* 
]W
VARIABLE_VALUElayer2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

10
21*

10
21*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUElayer3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

90
:1*

90
:1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUElayer4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

A0
B1*

A0
B1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEoutput/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEoutput/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

I0
J1*

I0
J1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEoutputlayer/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEoutputlayer/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

Q0
R1*

Q0
R1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
YS
VARIABLE_VALUE	sd/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEsd/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

Y0
Z1*

Y0
Z1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
c]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
C
0
1
2
3
4
5
6
7
	8*

�0*
* 
* 
* 
* 
* 
* 
�
e0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17*
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
�	variables
�	keras_api

�total

�count*
c]
VARIABLE_VALUEAdam/m/inputlayer/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/inputlayer/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/inputlayer/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/inputlayer/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/layer1/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/layer1/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/layer1/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/layer1/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/layer2/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/layer2/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/layer2/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/layer2/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/layer3/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/layer3/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/layer3/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/layer3/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/layer4/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/layer4/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/layer4/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/layer4/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/output/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/output/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/output/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/output/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/outputlayer/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/outputlayer/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/outputlayer/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/outputlayer/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/sd/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/sd/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/m/sd/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/v/sd/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/m/output_layer/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/output_layer/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/output_layer/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/output_layer/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
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
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameinputlayer/kernelinputlayer/biaslayer1/kernellayer1/biaslayer2/kernellayer2/biaslayer3/kernellayer3/biaslayer4/kernellayer4/biasoutput/kerneloutput/biasoutputlayer/kerneloutputlayer/bias	sd/kernelsd/biasoutput_layer/kerneloutput_layer/bias	iterationlearning_rateAdam/m/inputlayer/kernelAdam/v/inputlayer/kernelAdam/m/inputlayer/biasAdam/v/inputlayer/biasAdam/m/layer1/kernelAdam/v/layer1/kernelAdam/m/layer1/biasAdam/v/layer1/biasAdam/m/layer2/kernelAdam/v/layer2/kernelAdam/m/layer2/biasAdam/v/layer2/biasAdam/m/layer3/kernelAdam/v/layer3/kernelAdam/m/layer3/biasAdam/v/layer3/biasAdam/m/layer4/kernelAdam/v/layer4/kernelAdam/m/layer4/biasAdam/v/layer4/biasAdam/m/output/kernelAdam/v/output/kernelAdam/m/output/biasAdam/v/output/biasAdam/m/outputlayer/kernelAdam/v/outputlayer/kernelAdam/m/outputlayer/biasAdam/v/outputlayer/biasAdam/m/sd/kernelAdam/v/sd/kernelAdam/m/sd/biasAdam/v/sd/biasAdam/m/output_layer/kernelAdam/v/output_layer/kernelAdam/m/output_layer/biasAdam/v/output_layer/biastotalcountConst*G
Tin@
>2<*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_203022
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameinputlayer/kernelinputlayer/biaslayer1/kernellayer1/biaslayer2/kernellayer2/biaslayer3/kernellayer3/biaslayer4/kernellayer4/biasoutput/kerneloutput/biasoutputlayer/kerneloutputlayer/bias	sd/kernelsd/biasoutput_layer/kerneloutput_layer/bias	iterationlearning_rateAdam/m/inputlayer/kernelAdam/v/inputlayer/kernelAdam/m/inputlayer/biasAdam/v/inputlayer/biasAdam/m/layer1/kernelAdam/v/layer1/kernelAdam/m/layer1/biasAdam/v/layer1/biasAdam/m/layer2/kernelAdam/v/layer2/kernelAdam/m/layer2/biasAdam/v/layer2/biasAdam/m/layer3/kernelAdam/v/layer3/kernelAdam/m/layer3/biasAdam/v/layer3/biasAdam/m/layer4/kernelAdam/v/layer4/kernelAdam/m/layer4/biasAdam/v/layer4/biasAdam/m/output/kernelAdam/v/output/kernelAdam/m/output/biasAdam/v/output/biasAdam/m/outputlayer/kernelAdam/v/outputlayer/kernelAdam/m/outputlayer/biasAdam/v/outputlayer/biasAdam/m/sd/kernelAdam/v/sd/kernelAdam/m/sd/biasAdam/v/sd/biasAdam/m/output_layer/kernelAdam/v/output_layer/kernelAdam/m/output_layer/biasAdam/v/output_layer/biastotalcount*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_203205�	
�3
�
F__inference_sequential_layer_call_and_return_conditional_losses_202274
inputlayer_input#
inputlayer_202228:
inputlayer_202230: 
layer1_202233:	�
layer1_202235:	� 
layer2_202238:	�@
layer2_202240:@
layer3_202243:@ 
layer3_202245: 
layer4_202248: 
layer4_202250:
output_202253:
output_202255:$
outputlayer_202258: 
outputlayer_202260:
	sd_202263:
	sd_202265:%
output_layer_202268:!
output_layer_202270:
identity��"inputlayer/StatefulPartitionedCall�layer1/StatefulPartitionedCall�layer2/StatefulPartitionedCall�layer3/StatefulPartitionedCall�layer4/StatefulPartitionedCall�output/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�#outputlayer/StatefulPartitionedCall�sd/StatefulPartitionedCall�
"inputlayer/StatefulPartitionedCallStatefulPartitionedCallinputlayer_inputinputlayer_202228inputlayer_202230*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_inputlayer_layer_call_and_return_conditional_losses_202098�
layer1/StatefulPartitionedCallStatefulPartitionedCall+inputlayer/StatefulPartitionedCall:output:0layer1_202233layer1_202235*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_202113�
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_202238layer2_202240*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_202128�
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_202243layer3_202245*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_202143�
layer4/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0layer4_202248layer4_202250*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_layer4_layer_call_and_return_conditional_losses_202158�
output/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0output_202253output_202255*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_202173�
#outputlayer/StatefulPartitionedCallStatefulPartitionedCall'output/StatefulPartitionedCall:output:0outputlayer_202258outputlayer_202260*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_outputlayer_layer_call_and_return_conditional_losses_202188�
sd/StatefulPartitionedCallStatefulPartitionedCall,outputlayer/StatefulPartitionedCall:output:0	sd_202263	sd_202265*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_sd_layer_call_and_return_conditional_losses_202203�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall#sd/StatefulPartitionedCall:output:0output_layer_202268output_layer_202270*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_202218|
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^inputlayer/StatefulPartitionedCall^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall^layer4/StatefulPartitionedCall^output/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall$^outputlayer/StatefulPartitionedCall^sd/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : : : : : 2H
"inputlayer/StatefulPartitionedCall"inputlayer/StatefulPartitionedCall2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2J
#outputlayer/StatefulPartitionedCall#outputlayer/StatefulPartitionedCall28
sd/StatefulPartitionedCallsd/StatefulPartitionedCall:&"
 
_user_specified_name202270:&"
 
_user_specified_name202268:&"
 
_user_specified_name202265:&"
 
_user_specified_name202263:&"
 
_user_specified_name202260:&"
 
_user_specified_name202258:&"
 
_user_specified_name202255:&"
 
_user_specified_name202253:&
"
 
_user_specified_name202250:&	"
 
_user_specified_name202248:&"
 
_user_specified_name202245:&"
 
_user_specified_name202243:&"
 
_user_specified_name202240:&"
 
_user_specified_name202238:&"
 
_user_specified_name202235:&"
 
_user_specified_name202233:&"
 
_user_specified_name202230:&"
 
_user_specified_name202228:Y U
'
_output_shapes
:���������
*
_user_specified_nameinputlayer_input
�	
�
B__inference_layer1_layer_call_and_return_conditional_losses_202519

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
>__inference_sd_layer_call_and_return_conditional_losses_202633

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
H__inference_output_layer_layer_call_and_return_conditional_losses_202218

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
B__inference_output_layer_call_and_return_conditional_losses_202595

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_inputlayer_layer_call_fn_202490

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_inputlayer_layer_call_and_return_conditional_losses_202098o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name202486:&"
 
_user_specified_name202484:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
,__inference_outputlayer_layer_call_fn_202604

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_outputlayer_layer_call_and_return_conditional_losses_202188o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name202600:&"
 
_user_specified_name202598:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
B__inference_layer1_layer_call_and_return_conditional_losses_202113

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�3
�
F__inference_sequential_layer_call_and_return_conditional_losses_202225
inputlayer_input#
inputlayer_202099:
inputlayer_202101: 
layer1_202114:	�
layer1_202116:	� 
layer2_202129:	�@
layer2_202131:@
layer3_202144:@ 
layer3_202146: 
layer4_202159: 
layer4_202161:
output_202174:
output_202176:$
outputlayer_202189: 
outputlayer_202191:
	sd_202204:
	sd_202206:%
output_layer_202219:!
output_layer_202221:
identity��"inputlayer/StatefulPartitionedCall�layer1/StatefulPartitionedCall�layer2/StatefulPartitionedCall�layer3/StatefulPartitionedCall�layer4/StatefulPartitionedCall�output/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�#outputlayer/StatefulPartitionedCall�sd/StatefulPartitionedCall�
"inputlayer/StatefulPartitionedCallStatefulPartitionedCallinputlayer_inputinputlayer_202099inputlayer_202101*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_inputlayer_layer_call_and_return_conditional_losses_202098�
layer1/StatefulPartitionedCallStatefulPartitionedCall+inputlayer/StatefulPartitionedCall:output:0layer1_202114layer1_202116*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_202113�
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_202129layer2_202131*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_202128�
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_202144layer3_202146*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_202143�
layer4/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0layer4_202159layer4_202161*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_layer4_layer_call_and_return_conditional_losses_202158�
output/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0output_202174output_202176*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_202173�
#outputlayer/StatefulPartitionedCallStatefulPartitionedCall'output/StatefulPartitionedCall:output:0outputlayer_202189outputlayer_202191*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_outputlayer_layer_call_and_return_conditional_losses_202188�
sd/StatefulPartitionedCallStatefulPartitionedCall,outputlayer/StatefulPartitionedCall:output:0	sd_202204	sd_202206*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_sd_layer_call_and_return_conditional_losses_202203�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall#sd/StatefulPartitionedCall:output:0output_layer_202219output_layer_202221*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_202218|
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^inputlayer/StatefulPartitionedCall^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall^layer4/StatefulPartitionedCall^output/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall$^outputlayer/StatefulPartitionedCall^sd/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : : : : : 2H
"inputlayer/StatefulPartitionedCall"inputlayer/StatefulPartitionedCall2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2J
#outputlayer/StatefulPartitionedCall#outputlayer/StatefulPartitionedCall28
sd/StatefulPartitionedCallsd/StatefulPartitionedCall:&"
 
_user_specified_name202221:&"
 
_user_specified_name202219:&"
 
_user_specified_name202206:&"
 
_user_specified_name202204:&"
 
_user_specified_name202191:&"
 
_user_specified_name202189:&"
 
_user_specified_name202176:&"
 
_user_specified_name202174:&
"
 
_user_specified_name202161:&	"
 
_user_specified_name202159:&"
 
_user_specified_name202146:&"
 
_user_specified_name202144:&"
 
_user_specified_name202131:&"
 
_user_specified_name202129:&"
 
_user_specified_name202116:&"
 
_user_specified_name202114:&"
 
_user_specified_name202101:&"
 
_user_specified_name202099:Y U
'
_output_shapes
:���������
*
_user_specified_nameinputlayer_input
�	
�
B__inference_layer3_layer_call_and_return_conditional_losses_202143

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_202481
inputlayer_input
unknown:
	unknown_0:
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputlayer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_202086o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name202477:&"
 
_user_specified_name202475:&"
 
_user_specified_name202473:&"
 
_user_specified_name202471:&"
 
_user_specified_name202469:&"
 
_user_specified_name202467:&"
 
_user_specified_name202465:&"
 
_user_specified_name202463:&
"
 
_user_specified_name202461:&	"
 
_user_specified_name202459:&"
 
_user_specified_name202457:&"
 
_user_specified_name202455:&"
 
_user_specified_name202453:&"
 
_user_specified_name202451:&"
 
_user_specified_name202449:&"
 
_user_specified_name202447:&"
 
_user_specified_name202445:&"
 
_user_specified_name202443:Y U
'
_output_shapes
:���������
*
_user_specified_nameinputlayer_input
�
�
'__inference_layer4_layer_call_fn_202566

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_layer4_layer_call_and_return_conditional_losses_202158o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name202562:&"
 
_user_specified_name202560:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
'__inference_output_layer_call_fn_202585

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_202173o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name202581:&"
 
_user_specified_name202579:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
F__inference_inputlayer_layer_call_and_return_conditional_losses_202500

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
B__inference_layer4_layer_call_and_return_conditional_losses_202576

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
#__inference_sd_layer_call_fn_202623

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_sd_layer_call_and_return_conditional_losses_202203o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name202619:&"
 
_user_specified_name202617:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_layer2_layer_call_fn_202528

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_202128o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name202524:&"
 
_user_specified_name202522:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_layer3_layer_call_fn_202547

inputs
unknown:@ 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_202143o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name202543:&"
 
_user_specified_name202541:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
B__inference_layer3_layer_call_and_return_conditional_losses_202557

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
>__inference_sd_layer_call_and_return_conditional_losses_202203

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_output_layer_layer_call_fn_202642

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_202218o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name202638:&"
 
_user_specified_name202636:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
B__inference_layer4_layer_call_and_return_conditional_losses_202158

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�Z
�
!__inference__wrapped_model_202086
inputlayer_inputF
4sequential_inputlayer_matmul_readvariableop_resource:C
5sequential_inputlayer_biasadd_readvariableop_resource:C
0sequential_layer1_matmul_readvariableop_resource:	�@
1sequential_layer1_biasadd_readvariableop_resource:	�C
0sequential_layer2_matmul_readvariableop_resource:	�@?
1sequential_layer2_biasadd_readvariableop_resource:@B
0sequential_layer3_matmul_readvariableop_resource:@ ?
1sequential_layer3_biasadd_readvariableop_resource: B
0sequential_layer4_matmul_readvariableop_resource: ?
1sequential_layer4_biasadd_readvariableop_resource:B
0sequential_output_matmul_readvariableop_resource:?
1sequential_output_biasadd_readvariableop_resource:G
5sequential_outputlayer_matmul_readvariableop_resource:D
6sequential_outputlayer_biasadd_readvariableop_resource:>
,sequential_sd_matmul_readvariableop_resource:;
-sequential_sd_biasadd_readvariableop_resource:H
6sequential_output_layer_matmul_readvariableop_resource:E
7sequential_output_layer_biasadd_readvariableop_resource:
identity��,sequential/inputlayer/BiasAdd/ReadVariableOp�+sequential/inputlayer/MatMul/ReadVariableOp�(sequential/layer1/BiasAdd/ReadVariableOp�'sequential/layer1/MatMul/ReadVariableOp�(sequential/layer2/BiasAdd/ReadVariableOp�'sequential/layer2/MatMul/ReadVariableOp�(sequential/layer3/BiasAdd/ReadVariableOp�'sequential/layer3/MatMul/ReadVariableOp�(sequential/layer4/BiasAdd/ReadVariableOp�'sequential/layer4/MatMul/ReadVariableOp�(sequential/output/BiasAdd/ReadVariableOp�'sequential/output/MatMul/ReadVariableOp�.sequential/output_layer/BiasAdd/ReadVariableOp�-sequential/output_layer/MatMul/ReadVariableOp�-sequential/outputlayer/BiasAdd/ReadVariableOp�,sequential/outputlayer/MatMul/ReadVariableOp�$sequential/sd/BiasAdd/ReadVariableOp�#sequential/sd/MatMul/ReadVariableOp�
+sequential/inputlayer/MatMul/ReadVariableOpReadVariableOp4sequential_inputlayer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential/inputlayer/MatMulMatMulinputlayer_input3sequential/inputlayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,sequential/inputlayer/BiasAdd/ReadVariableOpReadVariableOp5sequential_inputlayer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/inputlayer/BiasAddBiasAdd&sequential/inputlayer/MatMul:product:04sequential/inputlayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'sequential/layer1/MatMul/ReadVariableOpReadVariableOp0sequential_layer1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential/layer1/MatMulMatMul&sequential/inputlayer/BiasAdd:output:0/sequential/layer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
(sequential/layer1/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/layer1/BiasAddBiasAdd"sequential/layer1/MatMul:product:00sequential/layer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'sequential/layer2/MatMul/ReadVariableOpReadVariableOp0sequential_layer2_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
sequential/layer2/MatMulMatMul"sequential/layer1/BiasAdd:output:0/sequential/layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
(sequential/layer2/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential/layer2/BiasAddBiasAdd"sequential/layer2/MatMul:product:00sequential/layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
'sequential/layer3/MatMul/ReadVariableOpReadVariableOp0sequential_layer3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
sequential/layer3/MatMulMatMul"sequential/layer2/BiasAdd:output:0/sequential/layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(sequential/layer3/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential/layer3/BiasAddBiasAdd"sequential/layer3/MatMul:product:00sequential/layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'sequential/layer4/MatMul/ReadVariableOpReadVariableOp0sequential_layer4_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
sequential/layer4/MatMulMatMul"sequential/layer3/BiasAdd:output:0/sequential/layer4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(sequential/layer4/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/layer4/BiasAddBiasAdd"sequential/layer4/MatMul:product:00sequential/layer4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'sequential/output/MatMul/ReadVariableOpReadVariableOp0sequential_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential/output/MatMulMatMul"sequential/layer4/BiasAdd:output:0/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(sequential/output/BiasAdd/ReadVariableOpReadVariableOp1sequential_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/output/BiasAddBiasAdd"sequential/output/MatMul:product:00sequential/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,sequential/outputlayer/MatMul/ReadVariableOpReadVariableOp5sequential_outputlayer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential/outputlayer/MatMulMatMul"sequential/output/BiasAdd:output:04sequential/outputlayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-sequential/outputlayer/BiasAdd/ReadVariableOpReadVariableOp6sequential_outputlayer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/outputlayer/BiasAddBiasAdd'sequential/outputlayer/MatMul:product:05sequential/outputlayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#sequential/sd/MatMul/ReadVariableOpReadVariableOp,sequential_sd_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential/sd/MatMulMatMul'sequential/outputlayer/BiasAdd:output:0+sequential/sd/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$sequential/sd/BiasAdd/ReadVariableOpReadVariableOp-sequential_sd_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/sd/BiasAddBiasAddsequential/sd/MatMul:product:0,sequential/sd/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-sequential/output_layer/MatMul/ReadVariableOpReadVariableOp6sequential_output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential/output_layer/MatMulMatMulsequential/sd/BiasAdd:output:05sequential/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
.sequential/output_layer/BiasAdd/ReadVariableOpReadVariableOp7sequential_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/output_layer/BiasAddBiasAdd(sequential/output_layer/MatMul:product:06sequential/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w
IdentityIdentity(sequential/output_layer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp-^sequential/inputlayer/BiasAdd/ReadVariableOp,^sequential/inputlayer/MatMul/ReadVariableOp)^sequential/layer1/BiasAdd/ReadVariableOp(^sequential/layer1/MatMul/ReadVariableOp)^sequential/layer2/BiasAdd/ReadVariableOp(^sequential/layer2/MatMul/ReadVariableOp)^sequential/layer3/BiasAdd/ReadVariableOp(^sequential/layer3/MatMul/ReadVariableOp)^sequential/layer4/BiasAdd/ReadVariableOp(^sequential/layer4/MatMul/ReadVariableOp)^sequential/output/BiasAdd/ReadVariableOp(^sequential/output/MatMul/ReadVariableOp/^sequential/output_layer/BiasAdd/ReadVariableOp.^sequential/output_layer/MatMul/ReadVariableOp.^sequential/outputlayer/BiasAdd/ReadVariableOp-^sequential/outputlayer/MatMul/ReadVariableOp%^sequential/sd/BiasAdd/ReadVariableOp$^sequential/sd/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : : : : : 2\
,sequential/inputlayer/BiasAdd/ReadVariableOp,sequential/inputlayer/BiasAdd/ReadVariableOp2Z
+sequential/inputlayer/MatMul/ReadVariableOp+sequential/inputlayer/MatMul/ReadVariableOp2T
(sequential/layer1/BiasAdd/ReadVariableOp(sequential/layer1/BiasAdd/ReadVariableOp2R
'sequential/layer1/MatMul/ReadVariableOp'sequential/layer1/MatMul/ReadVariableOp2T
(sequential/layer2/BiasAdd/ReadVariableOp(sequential/layer2/BiasAdd/ReadVariableOp2R
'sequential/layer2/MatMul/ReadVariableOp'sequential/layer2/MatMul/ReadVariableOp2T
(sequential/layer3/BiasAdd/ReadVariableOp(sequential/layer3/BiasAdd/ReadVariableOp2R
'sequential/layer3/MatMul/ReadVariableOp'sequential/layer3/MatMul/ReadVariableOp2T
(sequential/layer4/BiasAdd/ReadVariableOp(sequential/layer4/BiasAdd/ReadVariableOp2R
'sequential/layer4/MatMul/ReadVariableOp'sequential/layer4/MatMul/ReadVariableOp2T
(sequential/output/BiasAdd/ReadVariableOp(sequential/output/BiasAdd/ReadVariableOp2R
'sequential/output/MatMul/ReadVariableOp'sequential/output/MatMul/ReadVariableOp2`
.sequential/output_layer/BiasAdd/ReadVariableOp.sequential/output_layer/BiasAdd/ReadVariableOp2^
-sequential/output_layer/MatMul/ReadVariableOp-sequential/output_layer/MatMul/ReadVariableOp2^
-sequential/outputlayer/BiasAdd/ReadVariableOp-sequential/outputlayer/BiasAdd/ReadVariableOp2\
,sequential/outputlayer/MatMul/ReadVariableOp,sequential/outputlayer/MatMul/ReadVariableOp2L
$sequential/sd/BiasAdd/ReadVariableOp$sequential/sd/BiasAdd/ReadVariableOp2J
#sequential/sd/MatMul/ReadVariableOp#sequential/sd/MatMul/ReadVariableOp:($
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
resource:Y U
'
_output_shapes
:���������
*
_user_specified_nameinputlayer_input
�	
�
F__inference_inputlayer_layer_call_and_return_conditional_losses_202098

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
G__inference_outputlayer_layer_call_and_return_conditional_losses_202188

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_sequential_layer_call_fn_202315
inputlayer_input
unknown:
	unknown_0:
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputlayer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_202225o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name202311:&"
 
_user_specified_name202309:&"
 
_user_specified_name202307:&"
 
_user_specified_name202305:&"
 
_user_specified_name202303:&"
 
_user_specified_name202301:&"
 
_user_specified_name202299:&"
 
_user_specified_name202297:&
"
 
_user_specified_name202295:&	"
 
_user_specified_name202293:&"
 
_user_specified_name202291:&"
 
_user_specified_name202289:&"
 
_user_specified_name202287:&"
 
_user_specified_name202285:&"
 
_user_specified_name202283:&"
 
_user_specified_name202281:&"
 
_user_specified_name202279:&"
 
_user_specified_name202277:Y U
'
_output_shapes
:���������
*
_user_specified_nameinputlayer_input
�	
�
H__inference_output_layer_layer_call_and_return_conditional_losses_202652

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
B__inference_layer2_layer_call_and_return_conditional_losses_202128

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
B__inference_layer2_layer_call_and_return_conditional_losses_202538

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
G__inference_outputlayer_layer_call_and_return_conditional_losses_202614

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
B__inference_output_layer_call_and_return_conditional_losses_202173

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�#
"__inference__traced_restore_203205
file_prefix4
"assignvariableop_inputlayer_kernel:0
"assignvariableop_1_inputlayer_bias:3
 assignvariableop_2_layer1_kernel:	�-
assignvariableop_3_layer1_bias:	�3
 assignvariableop_4_layer2_kernel:	�@,
assignvariableop_5_layer2_bias:@2
 assignvariableop_6_layer3_kernel:@ ,
assignvariableop_7_layer3_bias: 2
 assignvariableop_8_layer4_kernel: ,
assignvariableop_9_layer4_bias:3
!assignvariableop_10_output_kernel:-
assignvariableop_11_output_bias:8
&assignvariableop_12_outputlayer_kernel:2
$assignvariableop_13_outputlayer_bias:/
assignvariableop_14_sd_kernel:)
assignvariableop_15_sd_bias:9
'assignvariableop_16_output_layer_kernel:3
%assignvariableop_17_output_layer_bias:'
assignvariableop_18_iteration:	 +
!assignvariableop_19_learning_rate: >
,assignvariableop_20_adam_m_inputlayer_kernel:>
,assignvariableop_21_adam_v_inputlayer_kernel:8
*assignvariableop_22_adam_m_inputlayer_bias:8
*assignvariableop_23_adam_v_inputlayer_bias:;
(assignvariableop_24_adam_m_layer1_kernel:	�;
(assignvariableop_25_adam_v_layer1_kernel:	�5
&assignvariableop_26_adam_m_layer1_bias:	�5
&assignvariableop_27_adam_v_layer1_bias:	�;
(assignvariableop_28_adam_m_layer2_kernel:	�@;
(assignvariableop_29_adam_v_layer2_kernel:	�@4
&assignvariableop_30_adam_m_layer2_bias:@4
&assignvariableop_31_adam_v_layer2_bias:@:
(assignvariableop_32_adam_m_layer3_kernel:@ :
(assignvariableop_33_adam_v_layer3_kernel:@ 4
&assignvariableop_34_adam_m_layer3_bias: 4
&assignvariableop_35_adam_v_layer3_bias: :
(assignvariableop_36_adam_m_layer4_kernel: :
(assignvariableop_37_adam_v_layer4_kernel: 4
&assignvariableop_38_adam_m_layer4_bias:4
&assignvariableop_39_adam_v_layer4_bias::
(assignvariableop_40_adam_m_output_kernel::
(assignvariableop_41_adam_v_output_kernel:4
&assignvariableop_42_adam_m_output_bias:4
&assignvariableop_43_adam_v_output_bias:?
-assignvariableop_44_adam_m_outputlayer_kernel:?
-assignvariableop_45_adam_v_outputlayer_kernel:9
+assignvariableop_46_adam_m_outputlayer_bias:9
+assignvariableop_47_adam_v_outputlayer_bias:6
$assignvariableop_48_adam_m_sd_kernel:6
$assignvariableop_49_adam_v_sd_kernel:0
"assignvariableop_50_adam_m_sd_bias:0
"assignvariableop_51_adam_v_sd_bias:@
.assignvariableop_52_adam_m_output_layer_kernel:@
.assignvariableop_53_adam_v_output_layer_kernel::
,assignvariableop_54_adam_m_output_layer_bias::
,assignvariableop_55_adam_v_output_layer_bias:#
assignvariableop_56_total: #
assignvariableop_57_count: 
identity_59��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*�
value�B�;B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*�
value�B~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*I
dtypes?
=2;	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp"assignvariableop_inputlayer_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp"assignvariableop_1_inputlayer_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp assignvariableop_2_layer1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_layer1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp assignvariableop_4_layer2_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_layer2_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp assignvariableop_6_layer3_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_layer3_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp assignvariableop_8_layer4_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_layer4_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp!assignvariableop_10_output_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_output_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_outputlayer_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp$assignvariableop_13_outputlayer_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_sd_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_sd_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp'assignvariableop_16_output_layer_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp%assignvariableop_17_output_layer_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_iterationIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_learning_rateIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp,assignvariableop_20_adam_m_inputlayer_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_v_inputlayer_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_m_inputlayer_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_v_inputlayer_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_m_layer1_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_v_layer1_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_m_layer1_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp&assignvariableop_27_adam_v_layer1_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_m_layer2_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_v_layer2_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_m_layer2_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp&assignvariableop_31_adam_v_layer2_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_m_layer3_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_v_layer3_kernelIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_m_layer3_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp&assignvariableop_35_adam_v_layer3_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_m_layer4_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_v_layer4_kernelIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_m_layer4_biasIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp&assignvariableop_39_adam_v_layer4_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_m_output_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_v_output_kernelIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp&assignvariableop_42_adam_m_output_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp&assignvariableop_43_adam_v_output_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp-assignvariableop_44_adam_m_outputlayer_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp-assignvariableop_45_adam_v_outputlayer_kernelIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp+assignvariableop_46_adam_m_outputlayer_biasIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_v_outputlayer_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp$assignvariableop_48_adam_m_sd_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp$assignvariableop_49_adam_v_sd_kernelIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp"assignvariableop_50_adam_m_sd_biasIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp"assignvariableop_51_adam_v_sd_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp.assignvariableop_52_adam_m_output_layer_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp.assignvariableop_53_adam_v_output_layer_kernelIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp,assignvariableop_54_adam_m_output_layer_biasIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp,assignvariableop_55_adam_v_output_layer_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOpassignvariableop_56_totalIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOpassignvariableop_57_countIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_58Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_59IdentityIdentity_58:output:0^NoOp_1*
T0*
_output_shapes
: �

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_59Identity_59:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
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
AssignVariableOp_57AssignVariableOp_572(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%:!

_user_specified_namecount:%9!

_user_specified_nametotal:884
2
_user_specified_nameAdam/v/output_layer/bias:874
2
_user_specified_nameAdam/m/output_layer/bias::66
4
_user_specified_nameAdam/v/output_layer/kernel::56
4
_user_specified_nameAdam/m/output_layer/kernel:.4*
(
_user_specified_nameAdam/v/sd/bias:.3*
(
_user_specified_nameAdam/m/sd/bias:02,
*
_user_specified_nameAdam/v/sd/kernel:01,
*
_user_specified_nameAdam/m/sd/kernel:703
1
_user_specified_nameAdam/v/outputlayer/bias:7/3
1
_user_specified_nameAdam/m/outputlayer/bias:9.5
3
_user_specified_nameAdam/v/outputlayer/kernel:9-5
3
_user_specified_nameAdam/m/outputlayer/kernel:2,.
,
_user_specified_nameAdam/v/output/bias:2+.
,
_user_specified_nameAdam/m/output/bias:4*0
.
_user_specified_nameAdam/v/output/kernel:4)0
.
_user_specified_nameAdam/m/output/kernel:2(.
,
_user_specified_nameAdam/v/layer4/bias:2'.
,
_user_specified_nameAdam/m/layer4/bias:4&0
.
_user_specified_nameAdam/v/layer4/kernel:4%0
.
_user_specified_nameAdam/m/layer4/kernel:2$.
,
_user_specified_nameAdam/v/layer3/bias:2#.
,
_user_specified_nameAdam/m/layer3/bias:4"0
.
_user_specified_nameAdam/v/layer3/kernel:4!0
.
_user_specified_nameAdam/m/layer3/kernel:2 .
,
_user_specified_nameAdam/v/layer2/bias:2.
,
_user_specified_nameAdam/m/layer2/bias:40
.
_user_specified_nameAdam/v/layer2/kernel:40
.
_user_specified_nameAdam/m/layer2/kernel:2.
,
_user_specified_nameAdam/v/layer1/bias:2.
,
_user_specified_nameAdam/m/layer1/bias:40
.
_user_specified_nameAdam/v/layer1/kernel:40
.
_user_specified_nameAdam/m/layer1/kernel:62
0
_user_specified_nameAdam/v/inputlayer/bias:62
0
_user_specified_nameAdam/m/inputlayer/bias:84
2
_user_specified_nameAdam/v/inputlayer/kernel:84
2
_user_specified_nameAdam/m/inputlayer/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:1-
+
_user_specified_nameoutput_layer/bias:3/
-
_user_specified_nameoutput_layer/kernel:'#
!
_user_specified_name	sd/bias:)%
#
_user_specified_name	sd/kernel:0,
*
_user_specified_nameoutputlayer/bias:2.
,
_user_specified_nameoutputlayer/kernel:+'
%
_user_specified_nameoutput/bias:-)
'
_user_specified_nameoutput/kernel:+
'
%
_user_specified_namelayer4/bias:-	)
'
_user_specified_namelayer4/kernel:+'
%
_user_specified_namelayer3/bias:-)
'
_user_specified_namelayer3/kernel:+'
%
_user_specified_namelayer2/bias:-)
'
_user_specified_namelayer2/kernel:+'
%
_user_specified_namelayer1/bias:-)
'
_user_specified_namelayer1/kernel:/+
)
_user_specified_nameinputlayer/bias:1-
+
_user_specified_nameinputlayer/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
'__inference_layer1_layer_call_fn_202509

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_202113p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name202505:&"
 
_user_specified_name202503:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�4
__inference__traced_save_203022
file_prefix:
(read_disablecopyonread_inputlayer_kernel:6
(read_1_disablecopyonread_inputlayer_bias:9
&read_2_disablecopyonread_layer1_kernel:	�3
$read_3_disablecopyonread_layer1_bias:	�9
&read_4_disablecopyonread_layer2_kernel:	�@2
$read_5_disablecopyonread_layer2_bias:@8
&read_6_disablecopyonread_layer3_kernel:@ 2
$read_7_disablecopyonread_layer3_bias: 8
&read_8_disablecopyonread_layer4_kernel: 2
$read_9_disablecopyonread_layer4_bias:9
'read_10_disablecopyonread_output_kernel:3
%read_11_disablecopyonread_output_bias:>
,read_12_disablecopyonread_outputlayer_kernel:8
*read_13_disablecopyonread_outputlayer_bias:5
#read_14_disablecopyonread_sd_kernel:/
!read_15_disablecopyonread_sd_bias:?
-read_16_disablecopyonread_output_layer_kernel:9
+read_17_disablecopyonread_output_layer_bias:-
#read_18_disablecopyonread_iteration:	 1
'read_19_disablecopyonread_learning_rate: D
2read_20_disablecopyonread_adam_m_inputlayer_kernel:D
2read_21_disablecopyonread_adam_v_inputlayer_kernel:>
0read_22_disablecopyonread_adam_m_inputlayer_bias:>
0read_23_disablecopyonread_adam_v_inputlayer_bias:A
.read_24_disablecopyonread_adam_m_layer1_kernel:	�A
.read_25_disablecopyonread_adam_v_layer1_kernel:	�;
,read_26_disablecopyonread_adam_m_layer1_bias:	�;
,read_27_disablecopyonread_adam_v_layer1_bias:	�A
.read_28_disablecopyonread_adam_m_layer2_kernel:	�@A
.read_29_disablecopyonread_adam_v_layer2_kernel:	�@:
,read_30_disablecopyonread_adam_m_layer2_bias:@:
,read_31_disablecopyonread_adam_v_layer2_bias:@@
.read_32_disablecopyonread_adam_m_layer3_kernel:@ @
.read_33_disablecopyonread_adam_v_layer3_kernel:@ :
,read_34_disablecopyonread_adam_m_layer3_bias: :
,read_35_disablecopyonread_adam_v_layer3_bias: @
.read_36_disablecopyonread_adam_m_layer4_kernel: @
.read_37_disablecopyonread_adam_v_layer4_kernel: :
,read_38_disablecopyonread_adam_m_layer4_bias::
,read_39_disablecopyonread_adam_v_layer4_bias:@
.read_40_disablecopyonread_adam_m_output_kernel:@
.read_41_disablecopyonread_adam_v_output_kernel::
,read_42_disablecopyonread_adam_m_output_bias::
,read_43_disablecopyonread_adam_v_output_bias:E
3read_44_disablecopyonread_adam_m_outputlayer_kernel:E
3read_45_disablecopyonread_adam_v_outputlayer_kernel:?
1read_46_disablecopyonread_adam_m_outputlayer_bias:?
1read_47_disablecopyonread_adam_v_outputlayer_bias:<
*read_48_disablecopyonread_adam_m_sd_kernel:<
*read_49_disablecopyonread_adam_v_sd_kernel:6
(read_50_disablecopyonread_adam_m_sd_bias:6
(read_51_disablecopyonread_adam_v_sd_bias:F
4read_52_disablecopyonread_adam_m_output_layer_kernel:F
4read_53_disablecopyonread_adam_v_output_layer_kernel:@
2read_54_disablecopyonread_adam_m_output_layer_bias:@
2read_55_disablecopyonread_adam_v_output_layer_bias:)
read_56_disablecopyonread_total: )
read_57_disablecopyonread_count: 
savev2_const
identity_117��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: z
Read/DisableCopyOnReadDisableCopyOnRead(read_disablecopyonread_inputlayer_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp(read_disablecopyonread_inputlayer_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:|
Read_1/DisableCopyOnReadDisableCopyOnRead(read_1_disablecopyonread_inputlayer_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp(read_1_disablecopyonread_inputlayer_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:z
Read_2/DisableCopyOnReadDisableCopyOnRead&read_2_disablecopyonread_layer1_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp&read_2_disablecopyonread_layer1_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	�x
Read_3/DisableCopyOnReadDisableCopyOnRead$read_3_disablecopyonread_layer1_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp$read_3_disablecopyonread_layer1_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:�z
Read_4/DisableCopyOnReadDisableCopyOnRead&read_4_disablecopyonread_layer2_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp&read_4_disablecopyonread_layer2_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0n

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@d

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@x
Read_5/DisableCopyOnReadDisableCopyOnRead$read_5_disablecopyonread_layer2_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp$read_5_disablecopyonread_layer2_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:@z
Read_6/DisableCopyOnReadDisableCopyOnRead&read_6_disablecopyonread_layer3_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp&read_6_disablecopyonread_layer3_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:@ x
Read_7/DisableCopyOnReadDisableCopyOnRead$read_7_disablecopyonread_layer3_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp$read_7_disablecopyonread_layer3_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: z
Read_8/DisableCopyOnReadDisableCopyOnRead&read_8_disablecopyonread_layer4_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp&read_8_disablecopyonread_layer4_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

: x
Read_9/DisableCopyOnReadDisableCopyOnRead$read_9_disablecopyonread_layer4_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp$read_9_disablecopyonread_layer4_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_10/DisableCopyOnReadDisableCopyOnRead'read_10_disablecopyonread_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp'read_10_disablecopyonread_output_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:z
Read_11/DisableCopyOnReadDisableCopyOnRead%read_11_disablecopyonread_output_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp%read_11_disablecopyonread_output_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_12/DisableCopyOnReadDisableCopyOnRead,read_12_disablecopyonread_outputlayer_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp,read_12_disablecopyonread_outputlayer_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_13/DisableCopyOnReadDisableCopyOnRead*read_13_disablecopyonread_outputlayer_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp*read_13_disablecopyonread_outputlayer_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_14/DisableCopyOnReadDisableCopyOnRead#read_14_disablecopyonread_sd_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp#read_14_disablecopyonread_sd_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:v
Read_15/DisableCopyOnReadDisableCopyOnRead!read_15_disablecopyonread_sd_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp!read_15_disablecopyonread_sd_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_16/DisableCopyOnReadDisableCopyOnRead-read_16_disablecopyonread_output_layer_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp-read_16_disablecopyonread_output_layer_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_17/DisableCopyOnReadDisableCopyOnRead+read_17_disablecopyonread_output_layer_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp+read_17_disablecopyonread_output_layer_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_18/DisableCopyOnReadDisableCopyOnRead#read_18_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp#read_18_disablecopyonread_iteration^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_19/DisableCopyOnReadDisableCopyOnRead'read_19_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp'read_19_disablecopyonread_learning_rate^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_20/DisableCopyOnReadDisableCopyOnRead2read_20_disablecopyonread_adam_m_inputlayer_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp2read_20_disablecopyonread_adam_m_inputlayer_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_21/DisableCopyOnReadDisableCopyOnRead2read_21_disablecopyonread_adam_v_inputlayer_kernel"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp2read_21_disablecopyonread_adam_v_inputlayer_kernel^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_22/DisableCopyOnReadDisableCopyOnRead0read_22_disablecopyonread_adam_m_inputlayer_bias"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp0read_22_disablecopyonread_adam_m_inputlayer_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_23/DisableCopyOnReadDisableCopyOnRead0read_23_disablecopyonread_adam_v_inputlayer_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp0read_23_disablecopyonread_adam_v_inputlayer_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_24/DisableCopyOnReadDisableCopyOnRead.read_24_disablecopyonread_adam_m_layer1_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp.read_24_disablecopyonread_adam_m_layer1_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_25/DisableCopyOnReadDisableCopyOnRead.read_25_disablecopyonread_adam_v_layer1_kernel"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp.read_25_disablecopyonread_adam_v_layer1_kernel^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_26/DisableCopyOnReadDisableCopyOnRead,read_26_disablecopyonread_adam_m_layer1_bias"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp,read_26_disablecopyonread_adam_m_layer1_bias^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_27/DisableCopyOnReadDisableCopyOnRead,read_27_disablecopyonread_adam_v_layer1_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp,read_27_disablecopyonread_adam_v_layer1_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_28/DisableCopyOnReadDisableCopyOnRead.read_28_disablecopyonread_adam_m_layer2_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp.read_28_disablecopyonread_adam_m_layer2_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@�
Read_29/DisableCopyOnReadDisableCopyOnRead.read_29_disablecopyonread_adam_v_layer2_kernel"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp.read_29_disablecopyonread_adam_v_layer2_kernel^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@�
Read_30/DisableCopyOnReadDisableCopyOnRead,read_30_disablecopyonread_adam_m_layer2_bias"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp,read_30_disablecopyonread_adam_m_layer2_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_31/DisableCopyOnReadDisableCopyOnRead,read_31_disablecopyonread_adam_v_layer2_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp,read_31_disablecopyonread_adam_v_layer2_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_32/DisableCopyOnReadDisableCopyOnRead.read_32_disablecopyonread_adam_m_layer3_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp.read_32_disablecopyonread_adam_m_layer3_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0o
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ e
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes

:@ �
Read_33/DisableCopyOnReadDisableCopyOnRead.read_33_disablecopyonread_adam_v_layer3_kernel"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp.read_33_disablecopyonread_adam_v_layer3_kernel^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0o
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ e
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes

:@ �
Read_34/DisableCopyOnReadDisableCopyOnRead,read_34_disablecopyonread_adam_m_layer3_bias"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp,read_34_disablecopyonread_adam_m_layer3_bias^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_35/DisableCopyOnReadDisableCopyOnRead,read_35_disablecopyonread_adam_v_layer3_bias"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp,read_35_disablecopyonread_adam_v_layer3_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_36/DisableCopyOnReadDisableCopyOnRead.read_36_disablecopyonread_adam_m_layer4_kernel"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp.read_36_disablecopyonread_adam_m_layer4_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_37/DisableCopyOnReadDisableCopyOnRead.read_37_disablecopyonread_adam_v_layer4_kernel"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp.read_37_disablecopyonread_adam_v_layer4_kernel^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_38/DisableCopyOnReadDisableCopyOnRead,read_38_disablecopyonread_adam_m_layer4_bias"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp,read_38_disablecopyonread_adam_m_layer4_bias^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_39/DisableCopyOnReadDisableCopyOnRead,read_39_disablecopyonread_adam_v_layer4_bias"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp,read_39_disablecopyonread_adam_v_layer4_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_40/DisableCopyOnReadDisableCopyOnRead.read_40_disablecopyonread_adam_m_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp.read_40_disablecopyonread_adam_m_output_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_41/DisableCopyOnReadDisableCopyOnRead.read_41_disablecopyonread_adam_v_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp.read_41_disablecopyonread_adam_v_output_kernel^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_42/DisableCopyOnReadDisableCopyOnRead,read_42_disablecopyonread_adam_m_output_bias"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp,read_42_disablecopyonread_adam_m_output_bias^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_43/DisableCopyOnReadDisableCopyOnRead,read_43_disablecopyonread_adam_v_output_bias"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp,read_43_disablecopyonread_adam_v_output_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_44/DisableCopyOnReadDisableCopyOnRead3read_44_disablecopyonread_adam_m_outputlayer_kernel"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp3read_44_disablecopyonread_adam_m_outputlayer_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_45/DisableCopyOnReadDisableCopyOnRead3read_45_disablecopyonread_adam_v_outputlayer_kernel"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp3read_45_disablecopyonread_adam_v_outputlayer_kernel^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_46/DisableCopyOnReadDisableCopyOnRead1read_46_disablecopyonread_adam_m_outputlayer_bias"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp1read_46_disablecopyonread_adam_m_outputlayer_bias^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_47/DisableCopyOnReadDisableCopyOnRead1read_47_disablecopyonread_adam_v_outputlayer_bias"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp1read_47_disablecopyonread_adam_v_outputlayer_bias^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_48/DisableCopyOnReadDisableCopyOnRead*read_48_disablecopyonread_adam_m_sd_kernel"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp*read_48_disablecopyonread_adam_m_sd_kernel^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_49/DisableCopyOnReadDisableCopyOnRead*read_49_disablecopyonread_adam_v_sd_kernel"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp*read_49_disablecopyonread_adam_v_sd_kernel^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes

:}
Read_50/DisableCopyOnReadDisableCopyOnRead(read_50_disablecopyonread_adam_m_sd_bias"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp(read_50_disablecopyonread_adam_m_sd_bias^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_51/DisableCopyOnReadDisableCopyOnRead(read_51_disablecopyonread_adam_v_sd_bias"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp(read_51_disablecopyonread_adam_v_sd_bias^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_52/DisableCopyOnReadDisableCopyOnRead4read_52_disablecopyonread_adam_m_output_layer_kernel"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp4read_52_disablecopyonread_adam_m_output_layer_kernel^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_53/DisableCopyOnReadDisableCopyOnRead4read_53_disablecopyonread_adam_v_output_layer_kernel"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp4read_53_disablecopyonread_adam_v_output_layer_kernel^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_54/DisableCopyOnReadDisableCopyOnRead2read_54_disablecopyonread_adam_m_output_layer_bias"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp2read_54_disablecopyonread_adam_m_output_layer_bias^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_55/DisableCopyOnReadDisableCopyOnRead2read_55_disablecopyonread_adam_v_output_layer_bias"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp2read_55_disablecopyonread_adam_v_output_layer_bias^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_56/DisableCopyOnReadDisableCopyOnReadread_56_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOpread_56_disablecopyonread_total^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_57/DisableCopyOnReadDisableCopyOnReadread_57_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOpread_57_disablecopyonread_count^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*�
value�B�;B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*�
value�B~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *I
dtypes?
=2;	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_116Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_117IdentityIdentity_116:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_117Identity_117:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesz
x: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
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
Read_57/ReadVariableOpRead_57/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=;9

_output_shapes
: 

_user_specified_nameConst:%:!

_user_specified_namecount:%9!

_user_specified_nametotal:884
2
_user_specified_nameAdam/v/output_layer/bias:874
2
_user_specified_nameAdam/m/output_layer/bias::66
4
_user_specified_nameAdam/v/output_layer/kernel::56
4
_user_specified_nameAdam/m/output_layer/kernel:.4*
(
_user_specified_nameAdam/v/sd/bias:.3*
(
_user_specified_nameAdam/m/sd/bias:02,
*
_user_specified_nameAdam/v/sd/kernel:01,
*
_user_specified_nameAdam/m/sd/kernel:703
1
_user_specified_nameAdam/v/outputlayer/bias:7/3
1
_user_specified_nameAdam/m/outputlayer/bias:9.5
3
_user_specified_nameAdam/v/outputlayer/kernel:9-5
3
_user_specified_nameAdam/m/outputlayer/kernel:2,.
,
_user_specified_nameAdam/v/output/bias:2+.
,
_user_specified_nameAdam/m/output/bias:4*0
.
_user_specified_nameAdam/v/output/kernel:4)0
.
_user_specified_nameAdam/m/output/kernel:2(.
,
_user_specified_nameAdam/v/layer4/bias:2'.
,
_user_specified_nameAdam/m/layer4/bias:4&0
.
_user_specified_nameAdam/v/layer4/kernel:4%0
.
_user_specified_nameAdam/m/layer4/kernel:2$.
,
_user_specified_nameAdam/v/layer3/bias:2#.
,
_user_specified_nameAdam/m/layer3/bias:4"0
.
_user_specified_nameAdam/v/layer3/kernel:4!0
.
_user_specified_nameAdam/m/layer3/kernel:2 .
,
_user_specified_nameAdam/v/layer2/bias:2.
,
_user_specified_nameAdam/m/layer2/bias:40
.
_user_specified_nameAdam/v/layer2/kernel:40
.
_user_specified_nameAdam/m/layer2/kernel:2.
,
_user_specified_nameAdam/v/layer1/bias:2.
,
_user_specified_nameAdam/m/layer1/bias:40
.
_user_specified_nameAdam/v/layer1/kernel:40
.
_user_specified_nameAdam/m/layer1/kernel:62
0
_user_specified_nameAdam/v/inputlayer/bias:62
0
_user_specified_nameAdam/m/inputlayer/bias:84
2
_user_specified_nameAdam/v/inputlayer/kernel:84
2
_user_specified_nameAdam/m/inputlayer/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:1-
+
_user_specified_nameoutput_layer/bias:3/
-
_user_specified_nameoutput_layer/kernel:'#
!
_user_specified_name	sd/bias:)%
#
_user_specified_name	sd/kernel:0,
*
_user_specified_nameoutputlayer/bias:2.
,
_user_specified_nameoutputlayer/kernel:+'
%
_user_specified_nameoutput/bias:-)
'
_user_specified_nameoutput/kernel:+
'
%
_user_specified_namelayer4/bias:-	)
'
_user_specified_namelayer4/kernel:+'
%
_user_specified_namelayer3/bias:-)
'
_user_specified_namelayer3/kernel:+'
%
_user_specified_namelayer2/bias:-)
'
_user_specified_namelayer2/kernel:+'
%
_user_specified_namelayer1/bias:-)
'
_user_specified_namelayer1/kernel:/+
)
_user_specified_nameinputlayer/bias:1-
+
_user_specified_nameinputlayer/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
+__inference_sequential_layer_call_fn_202356
inputlayer_input
unknown:
	unknown_0:
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputlayer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_202274o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name202352:&"
 
_user_specified_name202350:&"
 
_user_specified_name202348:&"
 
_user_specified_name202346:&"
 
_user_specified_name202344:&"
 
_user_specified_name202342:&"
 
_user_specified_name202340:&"
 
_user_specified_name202338:&
"
 
_user_specified_name202336:&	"
 
_user_specified_name202334:&"
 
_user_specified_name202332:&"
 
_user_specified_name202330:&"
 
_user_specified_name202328:&"
 
_user_specified_name202326:&"
 
_user_specified_name202324:&"
 
_user_specified_name202322:&"
 
_user_specified_name202320:&"
 
_user_specified_name202318:Y U
'
_output_shapes
:���������
*
_user_specified_nameinputlayer_input"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
M
inputlayer_input9
"serving_default_inputlayer_input:0���������@
output_layer0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
	layer_with_weights-8
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias"
_tf_keras_layer
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias"
_tf_keras_layer
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses

1kernel
2bias"
_tf_keras_layer
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias"
_tf_keras_layer
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses

Akernel
Bbias"
_tf_keras_layer
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

Ikernel
Jbias"
_tf_keras_layer
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias"
_tf_keras_layer
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias"
_tf_keras_layer
�
0
1
!2
"3
)4
*5
16
27
98
:9
A10
B11
I12
J13
Q14
R15
Y16
Z17"
trackable_list_wrapper
�
0
1
!2
"3
)4
*5
16
27
98
:9
A10
B11
I12
J13
Q14
R15
Y16
Z17"
trackable_list_wrapper
 "
trackable_list_wrapper
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
`trace_0
atrace_12�
+__inference_sequential_layer_call_fn_202315
+__inference_sequential_layer_call_fn_202356�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z`trace_0zatrace_1
�
btrace_0
ctrace_12�
F__inference_sequential_layer_call_and_return_conditional_losses_202225
F__inference_sequential_layer_call_and_return_conditional_losses_202274�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zbtrace_0zctrace_1
�B�
!__inference__wrapped_model_202086inputlayer_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
d
_variables
e_iterations
f_learning_rate
g_index_dict
h
_momentums
i_velocities
j_update_step_xla"
experimentalOptimizer
,
kserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
qtrace_02�
+__inference_inputlayer_layer_call_fn_202490�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zqtrace_0
�
rtrace_02�
F__inference_inputlayer_layer_call_and_return_conditional_losses_202500�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zrtrace_0
#:!2inputlayer/kernel
:2inputlayer/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
�
xtrace_02�
'__inference_layer1_layer_call_fn_202509�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zxtrace_0
�
ytrace_02�
B__inference_layer1_layer_call_and_return_conditional_losses_202519�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zytrace_0
 :	�2layer1/kernel
:�2layer1/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
�
trace_02�
'__inference_layer2_layer_call_fn_202528�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�
�trace_02�
B__inference_layer2_layer_call_and_return_conditional_losses_202538�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :	�@2layer2/kernel
:@2layer2/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_layer3_layer_call_fn_202547�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_layer3_layer_call_and_return_conditional_losses_202557�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:@ 2layer3/kernel
: 2layer3/bias
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_layer4_layer_call_fn_202566�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_layer4_layer_call_and_return_conditional_losses_202576�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
: 2layer4/kernel
:2layer4/bias
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_output_layer_call_fn_202585�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_output_layer_call_and_return_conditional_losses_202595�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:2output/kernel
:2output/bias
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_outputlayer_layer_call_fn_202604�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_outputlayer_layer_call_and_return_conditional_losses_202614�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
$:"2outputlayer/kernel
:2outputlayer/bias
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
#__inference_sd_layer_call_fn_202623�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
>__inference_sd_layer_call_and_return_conditional_losses_202633�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:2	sd/kernel
:2sd/bias
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_output_layer_layer_call_fn_202642�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_output_layer_layer_call_and_return_conditional_losses_202652�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
%:#2output_layer/kernel
:2output_layer/bias
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_sequential_layer_call_fn_202315inputlayer_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_sequential_layer_call_fn_202356inputlayer_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_sequential_layer_call_and_return_conditional_losses_202225inputlayer_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_sequential_layer_call_and_return_conditional_losses_202274inputlayer_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
e0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
$__inference_signature_wrapper_202481inputlayer_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_inputlayer_layer_call_fn_202490inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_inputlayer_layer_call_and_return_conditional_losses_202500inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
'__inference_layer1_layer_call_fn_202509inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_layer1_layer_call_and_return_conditional_losses_202519inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
'__inference_layer2_layer_call_fn_202528inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_layer2_layer_call_and_return_conditional_losses_202538inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
'__inference_layer3_layer_call_fn_202547inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_layer3_layer_call_and_return_conditional_losses_202557inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
'__inference_layer4_layer_call_fn_202566inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_layer4_layer_call_and_return_conditional_losses_202576inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
'__inference_output_layer_call_fn_202585inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_output_layer_call_and_return_conditional_losses_202595inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
,__inference_outputlayer_layer_call_fn_202604inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_outputlayer_layer_call_and_return_conditional_losses_202614inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
#__inference_sd_layer_call_fn_202623inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_sd_layer_call_and_return_conditional_losses_202633inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
-__inference_output_layer_layer_call_fn_202642inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_output_layer_layer_call_and_return_conditional_losses_202652inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
(:&2Adam/m/inputlayer/kernel
(:&2Adam/v/inputlayer/kernel
": 2Adam/m/inputlayer/bias
": 2Adam/v/inputlayer/bias
%:#	�2Adam/m/layer1/kernel
%:#	�2Adam/v/layer1/kernel
:�2Adam/m/layer1/bias
:�2Adam/v/layer1/bias
%:#	�@2Adam/m/layer2/kernel
%:#	�@2Adam/v/layer2/kernel
:@2Adam/m/layer2/bias
:@2Adam/v/layer2/bias
$:"@ 2Adam/m/layer3/kernel
$:"@ 2Adam/v/layer3/kernel
: 2Adam/m/layer3/bias
: 2Adam/v/layer3/bias
$:" 2Adam/m/layer4/kernel
$:" 2Adam/v/layer4/kernel
:2Adam/m/layer4/bias
:2Adam/v/layer4/bias
$:"2Adam/m/output/kernel
$:"2Adam/v/output/kernel
:2Adam/m/output/bias
:2Adam/v/output/bias
):'2Adam/m/outputlayer/kernel
):'2Adam/v/outputlayer/kernel
#:!2Adam/m/outputlayer/bias
#:!2Adam/v/outputlayer/bias
 :2Adam/m/sd/kernel
 :2Adam/v/sd/kernel
:2Adam/m/sd/bias
:2Adam/v/sd/bias
*:(2Adam/m/output_layer/kernel
*:(2Adam/v/output_layer/kernel
$:"2Adam/m/output_layer/bias
$:"2Adam/v/output_layer/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count�
!__inference__wrapped_model_202086�!")*129:ABIJQRYZ9�6
/�,
*�'
inputlayer_input���������
� ";�8
6
output_layer&�#
output_layer����������
F__inference_inputlayer_layer_call_and_return_conditional_losses_202500c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
+__inference_inputlayer_layer_call_fn_202490X/�,
%�"
 �
inputs���������
� "!�
unknown����������
B__inference_layer1_layer_call_and_return_conditional_losses_202519d!"/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
'__inference_layer1_layer_call_fn_202509Y!"/�,
%�"
 �
inputs���������
� ""�
unknown�����������
B__inference_layer2_layer_call_and_return_conditional_losses_202538d)*0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������@
� �
'__inference_layer2_layer_call_fn_202528Y)*0�-
&�#
!�
inputs����������
� "!�
unknown���������@�
B__inference_layer3_layer_call_and_return_conditional_losses_202557c12/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0��������� 
� �
'__inference_layer3_layer_call_fn_202547X12/�,
%�"
 �
inputs���������@
� "!�
unknown��������� �
B__inference_layer4_layer_call_and_return_conditional_losses_202576c9:/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������
� �
'__inference_layer4_layer_call_fn_202566X9:/�,
%�"
 �
inputs��������� 
� "!�
unknown����������
B__inference_output_layer_call_and_return_conditional_losses_202595cAB/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
'__inference_output_layer_call_fn_202585XAB/�,
%�"
 �
inputs���������
� "!�
unknown����������
H__inference_output_layer_layer_call_and_return_conditional_losses_202652cYZ/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
-__inference_output_layer_layer_call_fn_202642XYZ/�,
%�"
 �
inputs���������
� "!�
unknown����������
G__inference_outputlayer_layer_call_and_return_conditional_losses_202614cIJ/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
,__inference_outputlayer_layer_call_fn_202604XIJ/�,
%�"
 �
inputs���������
� "!�
unknown����������
>__inference_sd_layer_call_and_return_conditional_losses_202633cQR/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� 
#__inference_sd_layer_call_fn_202623XQR/�,
%�"
 �
inputs���������
� "!�
unknown����������
F__inference_sequential_layer_call_and_return_conditional_losses_202225�!")*129:ABIJQRYZA�>
7�4
*�'
inputlayer_input���������
p

 
� ",�)
"�
tensor_0���������
� �
F__inference_sequential_layer_call_and_return_conditional_losses_202274�!")*129:ABIJQRYZA�>
7�4
*�'
inputlayer_input���������
p 

 
� ",�)
"�
tensor_0���������
� �
+__inference_sequential_layer_call_fn_202315z!")*129:ABIJQRYZA�>
7�4
*�'
inputlayer_input���������
p

 
� "!�
unknown����������
+__inference_sequential_layer_call_fn_202356z!")*129:ABIJQRYZA�>
7�4
*�'
inputlayer_input���������
p 

 
� "!�
unknown����������
$__inference_signature_wrapper_202481�!")*129:ABIJQRYZM�J
� 
C�@
>
inputlayer_input*�'
inputlayer_input���������";�8
6
output_layer&�#
output_layer���������