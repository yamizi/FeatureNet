import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';

import Grid from '@material-ui/core/Grid';
import Paper from '@material-ui/core/Paper';
import Typography from '@material-ui/core/Typography';
import Fab from '@material-ui/core/Fab';
import SaveIcon from '@material-ui/icons/Save';



import 'rc-tree/assets/index.css';
import Tree, { TreeNode } from 'rc-tree';

import { defaultCell, defaultXML,buildTree } from '../util';

var fileDownload = require('js-file-download');

const styles = theme => ({
    avatar: {
      margin: 10,
    },
    root: {
      flexGrow: 1,
    },
    grow: {
      flexGrow: 1,
    },
    menuButton: {
      marginLeft: -12,
      marginRight: 20,
    },
    rootList: {
      width: '100%',
      
      backgroundColor: theme.palette.background.paper,
    },
    inline: {
      display: 'inline',
    },
    chip: {
      margin: theme.spacing.unit,
    },
    rightTaskInfo:{
      width:'200px'
    },
    fab: {
      margin: theme.spacing.unit,
      float:"right",
      left: theme.spacing.unit * 7,
    },
    fabDelete: {
      margin: theme.spacing.unit,
      position: 'fixed',
      bottom: theme.spacing.unit * 2,
      right: theme.spacing.unit * 10,
    },

    paper: {
        minHeight: 140,
        minWidth: 100,
        padding: theme.spacing(2),
      },
    
  });



class FMComponent extends React.Component {

    constructor(props) {
      // Required step: always call the parent class' constructor
      super(props);
  
      let date= new Date();
  
      // Set the state directly. Use props if necessary.
      
      this.state = {
        checkedKeys: [],  
        expandedKeys: ['cell'],
        autoExpandParent: true,
        selectedKeys: [],
        treeData: defaultCell,
        outputXML: [],
      }
  
    }

    onExpand = (expandedKeys) => {
        console.log('onExpand', expandedKeys);
        // if not set autoExpandParent to false, if children expanded, parent can not collapse.
        // or, you can remove all expanded chilren keys.
        this.setState({
          expandedKeys,
          autoExpandParent: false,
        });
      }
      onCheck = (checkedKeys) => {
        this.setState({
          checkedKeys,
        });
      }
      onSelect = (selectedKeys, info) => {
        console.log('onSelect', selectedKeys, info);
        this.setState({
          selectedKeys,
        });
      }

      handleConvertClick = () => {
       
        var output = buildTree(this.state.checkedKeys)
        fileDownload(output.join("\n"), 'fm.xml');

        this.setState({ outputXML: output});
      };

    render() {
        const { classes } = this.props;
        const {outputXML } = this.state

        const loop = data => {
            return data.map((item) => {
              if (item.children) {
                return (
                  <TreeNode
                    key={item.key} title={item.title}
                    disableCheckbox={item.disabled}
                  >
                    {loop(item.children)}
                  </TreeNode>
                );
              }
              return <TreeNode key={item.key} title={item.title} />;
            });
          };

        return (
            <div className={classes.root}>        
        
                <Grid container justify="center" spacing={3}>
                
                    <Grid item xs={10}>
                        
                        <Paper className={classes.paper}>
                            <Typography variant="h5" component="h3">
                                Cell features to configure:
                            </Typography>
                            <Fab color="primary" aria-label="Add" className={classes.fab} onClick={this.handleConvertClick}>
                                <SaveIcon />
                            </Fab>
                            <Typography component="p">
                                <Tree
                                    checkable
                                    onExpand={this.onExpand} expandedKeys={this.state.expandedKeys}
                                    autoExpandParent={this.state.autoExpandParent}
                                    onCheck={this.onCheck} checkedKeys={this.state.checkedKeys}
                                >
                                    {loop(this.state.treeData)}
                                </Tree>
                            </Typography>
                        </Paper>
                    </Grid>

                    {false && <Grid item xs={6}>
                        <Paper className={classes.paper}>
                            <Typography variant="h5" component="h3">
                                FeatureNet Feature Model:
                            </Typography>
                            <Typography component="p">

                            {this.state.outputXML.map(val => (
                                <Typography component="p">
                                    {val}
                                </Typography>
                            ))}
                        
                            </Typography>
                        </Paper>
                    </Grid>
                    }
               
                </Grid>
                

                
            </div>
        );
    }
}
    
FMComponent.propTypes = {
    classes: PropTypes.object.isRequired,
};


const FMPage = withStyles(styles)(FMComponent);

export default FMPage