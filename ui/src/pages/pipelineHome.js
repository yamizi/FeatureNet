import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import Button from '@material-ui/core/Button';
import TextField from '@material-ui/core/TextField';
import Grid from '@material-ui/core/Grid';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogTitle from '@material-ui/core/DialogTitle';
import FormLabel from '@material-ui/core/FormLabel';
import FormControl from '@material-ui/core/FormControl';

import Radio from '@material-ui/core/Radio';
import RadioGroup from '@material-ui/core/RadioGroup';

import API from '../api';


const styles  = theme => ( {
    root: {
      flexGrow: 1,
    },
    grow: {
      flexGrow: 1,
    },
    textField: {
      marginLeft: theme.spacing.unit,
      marginRight: theme.spacing.unit,
      width: "100%"
    },
    menuButton: {
      marginLeft: -12,
      marginRight: 20,
    },
    container: {
      display: 'flex',
      flexWrap: 'wrap',
    },
    dense: {
      marginTop: 19,
    },
    menu: {
      width: 200,
    },
    form: {
      flexDirection: 'column',  
      alignItems: 'left',
    }, 
    list: {
      display: 'flex',
      flexDirection: 'column',
      width: '100%',
    },
    formControl: {
      margin: theme.spacing.unit * 3,
    },
    marginTop: {
      marginTop: '20px',
      width: '100%'
    }
  });
  

class PipelineHomeComponent extends React.Component {

    constructor(props) {
        // Required step: always call the parent class' constructor
        super(props);

        // Set the state directly. Use props if necessary.
        this.state = {
            id:0,
            sampling_params:{
              task_name:"",
              dataset:"mnist",
              max_sampling_time:"30",
              nb_initial_config:"100",
              max_nb_cells:"5",
              max_nb_blocks:"5",
              nb_training_iterations:"12",
            },
            datasets:[
            {
                label:"CIFAR-10",
                value:"cifar"
            },
            {
                label:"MNIST",
                value:"mnist"
            }
            ],
            new_task_open : true

        }

    }

    
    

    render() {
        const { classes } = this.props;
        const {datasets, sampling_params} = this.state;
        
    
        return (
    
          <div>
            <Dialog
              open = "true"
              onClose={this.handleClose}
              aria-labelledby="form-dialog-title"
              maxWidth="md"
              fullWidth = "true"
            >
              <DialogTitle id="form-dialog-title">Diversity Sampling</DialogTitle>
              <DialogContent>
    
              <form className={classes.container} noValidate autoComplete="off">
              <Grid container className={classes.form}>
                <Grid item> 

                  <TextField
                    id="task_name"
                    label="Task name"
                    className={classes.textField}
                    onChange={this.handleSamplingParamsChange('task_name')}
                    margin="normal"
                    variant="outlined"
                    fullWidth = "true"
                    defaultValue={sampling_params.task_name}
                  />

                  <TextField
                    id="max_sampling_time"
                    label="Maximum sampling time (in seconds)"
                    className={classes.textField}
                    onChange={this.handleSamplingParamsChange('max_sampling_time')}
                    margin="normal"
                    variant="outlined"
                    type="number"
                    fullWidth = "true"
                    defaultValue={sampling_params.max_sampling_time}
                  />
                </Grid>
    
                <Grid item >
                  <TextField
                    id="nb_initial_config"
                    label="Number of sampled configuration"
                    onChange={this.handleSamplingParamsChange('nb_initial_config')}
                    className={classes.textField}
                    margin="normal"
                    variant="outlined"
                    type="number"
                    fullWidth = "true"
                    defaultValue={sampling_params.nb_initial_config}
                  />
                </Grid>
    
                <Grid item >
                  <TextField
                    id="max_nb_blocks"
                    label="Maximum number of blocks"
                    fullWidth="true"
                    onChange={this.handleSamplingParamsChange('max_nb_blocks')}
                    className={classes.textField}
                    margin="normal"
                    variant="outlined"
                    type="number"
                    defaultValue={sampling_params.max_nb_blocks}
                  />
                </Grid>
    
                <Grid item >
                  <TextField
                    id="max_nb_cells"
                    label="Maximum number of cells"
                    fullWidth="true"
                    onChange={this.handleSamplingParamsChange('max_nb_cells')}
                    className={classes.textField}
                    margin="normal"
                    variant="outlined"
                    type="number"
                    defaultValue={sampling_params.max_nb_cells}
                  />
                </Grid>


                <Grid item >
                  <TextField
                    id="nb_training_iterations"
                    label="Number of training iterations"
                    fullWidth="true"
                    onChange={this.handleSamplingParamsChange('nb_training_iterations')}
                    className={classes.textField}
                    margin="normal"
                    variant="outlined"
                    type="number"
                    defaultValue={sampling_params.nb_training_iterations}
                  />
                </Grid>
    
                <Grid item className={classes.list}>
                  <FormControl required component="fieldset" className={classes.formControl}>
                    <FormLabel component="legend">Datasets</FormLabel>
                    <RadioGroup aria-label="dataset" value={sampling_params.dataset} onChange={this.handleDatasetChange}>
                        {datasets.map((item) =>
                        <FormControlLabel value={item.value} control={<Radio />} label={item.label} />
                        )}
                    </RadioGroup>
                  </FormControl>
                </Grid>
              </Grid> 
            </form>
                
              </DialogContent>
              <DialogActions>
                <Button onClick={this.handleSubmit} color="primary">
                  Generate
                </Button>
              </DialogActions>
            </Dialog>
          </div>
        );
    }
    
    
    handleSamplingParamsChange = name => event => {
      var sampling_params = {...this.state.sampling_params}
      sampling_params[name] = event.target.value;
      this.setState({sampling_params})
    };

    handleDatasetChange = event => {
      var val = event.target.value
      var sampling_params = {...this.state.sampling_params}
      sampling_params.dataset = val;
      this.setState({sampling_params})

  };
    
    handleClose = (event) => {
      this.props.onClose(event)
    };

    handleSubmit = (event) =>{

      let data = this.state.sampling_params
          
      API.post(`sample/`, { data: data, auth: {userId:this.state.id} })
          .then(res => {
            this.handleClose(event)
          })
    }
    
    
}

PipelineHomeComponent.propTypes = {
    classes: PropTypes.object.isRequired
};


const PipelineHomePage = withStyles(styles)(PipelineHomeComponent);

export default PipelineHomePage