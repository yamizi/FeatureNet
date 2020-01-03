function conv_kernels(key){
    
    return [
        { title: `1x1`, key: `1x1-kernel`+key},
        { title: `3x1`, key: `1x1-kernel`+key},
        { title: `1x3`, key: `1x1-kernel`+key},
        { title: `3x3`, key: `3x3-kernel`+key},
        { title: `1x5`, key: `1x5-kernel`+key},
        { title: `5x1`, key: `5x1-kernel`+key},
        { title: `5x5`, key: `5x5-kernel`+key},
        { title: `7x1`, key: `7x1-kernel`+key},
        { title: `1x7`, key: `1x7-kernel`+key}
    ]
}

function pool_kernels(key){
    
    return [
        { title: `1x1`, key: `1x1-kernel`+key},
        { title: `2x2`, key: `2x2-kernel`+key},
        { title: `3x3`, key: `3x3-kernel`+key},
    ]
}

function conv_types(key){
    return[
        { title: `normal`, key: `normal-type`+key},
        { title: `separable`, key: `separable-type`+key},
        { title: `depthwise`, key: `depthwise-type`+key}
    ]
}

function pool_types(key){
    return[
        { title: `max`, key: `max-type`+key},
        { title: `average`, key: `average-type`+key},
        { title: `dilated`, key: `dilated-type`+key},
        { title: `global`, key: `global-type`+key}
    ]
}

function padding(key){
    return[
        { title: `same`, key: `same-padding`+key},
        { title: `valid`, key: `valid-padding`+key},
    ]
}


function activation(key){
    return[
        { title: `relu`, key: `relu-activation`+key},
        { title: `sigmoid`, key: `sigmoid-activation`+key, disabled:true},
        { title: `tanh`, key: `tanh-activation`+key, disabled:true},
        { title: `softmax`, key: `softmax-activation`+key, disabled:true}
    ]
}

function stride(key){
    return[
        { title: `1x1`, key: `1x1-stride`+key, disabled:true},
        { title: `2x2`, key: `2x2-stride`+key},
        { title: `3x3`, key: `3x3-stride`+key, disabled:true}
    ]
}

function dense_features(key){
    return[
        { title: `8`, key: `8-features`+key},
        { title: `16`, key: `16-features`+key},
        { title: `32`, key: `32-features`+key},
        { title: `64`, key: `64-features`+key},
        { title: `96`, key: `128-features`+key},
        { title: `256`, key: `256-features`+key, disabled:true},
        { title: `512`, key: `512-features`+key, disabled:true},
        { title: `1024`, key: `1024-features`+key, disabled:true},
        { title: `2048`, key: `2048-features`+key, disabled:true}
    ]
}

function conv_features(key){
    return[
        { title: `16`, key: `16-features`+key},
        { title: `32`, key: `32-features`+key},
        { title: `64`, key: `64-features`+key},
        { title: `128`, key: `128-features`+key},
        { title: `256`, key: `256-features`+key, disabled:true},
        { title: `512`, key: `512-features`+key, disabled:true},
        { title: `1024`, key: `1024-features`+key, disabled:true},
        { title: `2048`, key: `2048-features`+key, disabled:true}
    ]
}

function padding_fillSize(key){
    return[
        { title: `0x1`, key: `0x1-features`+key},
        { title: `1x0`, key: `1x0-features`+key},
        { title: `1x1`, key: `1x1-features`+key},
        { title: `3x3`, key: `3x3-features`+key}
    ]
}

function dropout_value(key){
    return[
        { title: `0`, key: `0-features`+key},
        { title: `2`, key: `2-features`+key},
        { title: `5`, key: `5-features`+key},
        { title: `7`, key: `7-features`+key}
    ]
}

function relativeCellIndex(key){
    return[
        { title: `0`, key: `0-relativeCellIndex`+key},
        { title: `1`, key: `1-relativeCellIndex`+key},
        { title: `2`, key: `2-relativeCellIndex`+key}
    ]
}

function operation(operationId)

{   return { title: `Operation`+operationId, key: `operation`+operationId+``, children:[
        { title: `Void`, key: `operation`+operationId+`-void`},
        { title: `BatchNormalization`, key: `batchnorm`+operationId+`-batchnorm`},
        { title: `Flatten`, key: `operation`+operationId+`-flatten`},
        { title: `Activation`, key: `activation`+operationId+`-identity`,children:activation(`operation`+operationId+`-activation`)},
        { title: `Padding`, key: `operation`+operationId+`-padding`, children:[
            { title: `fillSize`, key: `operation`+operationId+`-padding-fillSize`, children:padding_fillSize(`operation`+operationId+`-padding`)},
        ]},
        { title: `Dropout`, key: `operation`+operationId+`-dropout`, children:[
            { title: `value`, key: `operation`+operationId+`-dropout-value`, children:dropout_value(`operation`+operationId+`-dropout`)},
        ]},
    ,
    ]}
}


function combination()

{   return { title: `Combination`, key: `combination`+``, children:[
        { title: `Sum`, key: `combination`+`-sum`},
        { title: `Concat`, key: `combination`+`-concat`},
        
    ,
    ]}
}

function output()

{   return { title: `Output`, key: `output`+``, children:[
        { title: `Block`, key: `output`+`-block`},
        { title: `Cell`, key: `output`+`-cell`, children:[
            { title: `relativeCellIndex`, key: `output-cell-relativeCellIndex`, children:relativeCellIndex(`output-cell`)}

        ]}
    ]}
}

function input(inputId)

{   return { title: `Input`+inputId, key: `input`+inputId+``, children:[
        { title: `Convolution`, key: `input`+inputId+`-convolution`, children:[
            { title: `kernel`, key: `input`+inputId+`-convolution-kernel`, children:conv_kernels(`input`+inputId+`-convolution`)},
            { title: `type`, key: `input`+inputId+`-convolution-type`, children:conv_types(`input`+inputId+`-convolution`)},
            { title: `activation`, key: `input`+inputId+`-convolution-activation`, children:activation(`input`+inputId+`-convolution`)},
            { title: `padding`, key: `input`+inputId+`-convolution-padding`, children:padding(`input`+inputId+`-convolution`)},
            { title: `features`, key: `input`+inputId+`-convolution-features`, children:conv_features(`input`+inputId+`-convolution`)},
            { title: `stride`, key: `input`+inputId+`-convolution-stride`, children:stride(`input`+inputId+`-convolution`)},
            
        ]},
        { title: `Identity`, key: `input`+inputId+`-identity`},
        { title: `Recurrence`, key: `input`+inputId+`-recurrence`},
        { title: `Zeros`, key: `input`+inputId+`-zeroes`},
        { title: `Identity`, key: `input`+inputId+`-identity`},
        { title: `Pooling`, key: `input`+inputId+`-pooling`, children:[
            { title: `kernel`, key: `input`+inputId+`-pooling-kernel`, children:pool_kernels(`input`+inputId+`-pool`)},
            { title: `type`, key: `input`+inputId+`-pooling-type`, children:pool_types(`input`+inputId+`-pool`)},
            { title: `padding`, key: `input`+inputId+`-pooling-padding`, children:padding(`input`+inputId+`-pool`)},
            { title: `stride`, key: `input`+inputId+`-pooling-stride`, children:stride(`input`+inputId+`-pool`)},
            
        ]},
        { title: `Dense`, key: `input`+inputId+`-dense`, children:[
            { title: `features`, key: `input`+inputId+`-dense-features`, children:dense_features(`input`+inputId+`-dense`)},
            { title: `activation`, key: `input`+inputId+`-dense-activation`, children:activation(`input`+inputId+`-dense`)},
        
            ]
        },  
    ,
    ]}
}



export const defaultCell = [{ title: `Cell`, key: `cell`, children:[input(1),input(2),operation(1),operation(2), combination(), output()] }] 

export const defaultXML = [
    `
    <?xml version="1.0" encoding="UTF-8" standalone="no"?>
<feature_model name="FeatureNet model">
    <feature_tree>
:r Root(Root)
	:m Base(Base)
		:m Training(Training)
			:m Architecture(Architecture)
				:m Input(Input)
				:m Output(Output)
				:o Block[k](Block[k])
					:m Block[k]_stride(Block[k]_stride)
						:g [1,1]
							: Block[k]_stride_2x2(Block[k]_stride_2x2)
							: Block[k]_stride_1x1(Block[k]_stride_1x1)
					:m Block[k]_features(Block[k]_features)
						:g [1,1]
							: Block[k]_features_800(Block[k]_features_800)
							: Block[k]_features_400(Block[k]_features_400)
							: Block[k]_features_200(Block[k]_features_200)
							: Block[k]_features_100(Block[k]_features_100)
							: Block[k]_features_50(Block[k]_features_50)
							: Block[k]_features_25(Block[k]_features_25)
                    :o Block[k]_Element[i](Block[k]_Element[i])
                        :o Block[k]_Element[i]_Cell(Block[k]_Element[i]_Cell)
                    `,
                    'Block[k]_Element[i]_Cell'
                    ,
                    `</feature_tree>
                    <constraints>
                C1:~Architecture  or  Block1
                C2:~Block[k+1]  or  Block[k]
                C3:~Block[k]_Element[i+1]  or  Block[k]_Element[i]
                C4:~Block[k]_Element[i]_Cell_Output_Block  or  Block[k+1]
                C5:~Block[k]_Element[i]_Cell_Output_Block  or  ~Block[k]_Element[i+1]
                C6:~Block[k]_Element[i]_Cell_Output_Cell  or  Block[k]_Element[i+1]
                C7:~Architecture  or  ~Block[k]_Element[i]_Cell_Input1_Zeros
                
                </constraints>
                </feature_model>
                `   
]


var iterNode = function(node, selectedNodes,lbl_cat, output){
    if (node.key in selectedNodes){ 
        lbl_cat = lbl_cat +"_"+ node.title
        if(node.children){
            
            output.push(":m "+lbl_cat+"("+lbl_cat+")")
            output.push(":g [1,1]")
        }
        else{
            output.push(": "+lbl_cat+"("+lbl_cat+")")
        }

        console.log(output)

        if (node.children){
            for (var i in node.children){
                var n = node.children[i]
                iterNode(n, selectedNodes,lbl_cat, output)
            }
        }
        

    }
} 
export var buildTree = function(selectedNodes){
    
    selectedNodes = selectedNodes.reduce(function(map, obj) {
        map[obj] = true;
        return map;
    }, {});

    var output = [defaultXML[0]]
    
    var cell  = defaultCell[0]
    for (var i in cell.children){
        var n = cell.children[i]
        iterNode(n, selectedNodes, defaultXML[1],output)
    }

    output.push(defaultXML[2])
    
    return output
}

