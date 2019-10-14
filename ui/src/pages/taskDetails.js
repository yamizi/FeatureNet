import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogTitle from '@material-ui/core/DialogTitle';

import HomeIcon from '@material-ui/icons/Home';
import AppsIcon from '@material-ui/icons/Apps';
import ListIcon from '@material-ui/icons/List';

import Typography from '@material-ui/core/Typography';
import Box from '@material-ui/core/Box';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Switch from '@material-ui/core/Switch';

import Tabs from '@material-ui/core/Tabs';
import Tab from '@material-ui/core/Tab';

import API from '../api';

import TableModelsComponent from './modelsTable'
import ModelsDashboard from './modelsDashboard'


function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <Typography
      component="div"
      role="tabpanel"
      hidden={value !== index}
      id={`nav-tabpanel-${index}`}
      aria-labelledby={`nav-tab-${index}`}
      {...other}
    >
      <Box p={3}>{children}</Box>
    </Typography>
  );
}


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
    },
    switch:{
      marginLeft: theme.spacing.unit * 3,
    },
  });
  

class TaskDetailsComponent extends React.Component {

    constructor(props) {
        // Required step: always call the parent class' constructor
        super(props);

        // Set the state directly. Use props if necessary.
        this.state = {
            task: {},
            view_id:0
        }

        this.taskDetails(this.props.task.task_id)

    }

    async taskDetails(id){
      API.get(`sample/`+id, {params:{userId:this.props.usr.id, full:true}})
        .then(res => {
          this.setState({task: res.data})
      })
    }

    

    handleViewChange = (event, newValue) => {
      this.setState({view_id: newValue });
    };

    render() {
        const {task, view_id} = this.state;
        const { classes } = this.props;

        return (
    
          <div>
            
            <Dialog
              open = "true"
              onClose={this.handleClose}
              aria-labelledby="form-dialog-title"
              maxWidth="lg"
              fullWidth="true" 
            >
              <DialogTitle id="form-dialog-title">
                Details for: {task.task_name}
              </DialogTitle>
              <DialogContent>
                <Tabs
                  value={view_id}
                  onChange={this.handleViewChange}
                  variant="fullWidth"
                  indicatorColor="secondary"
                  textColor="secondary"
                  aria-label="icon label tabs example"
                >
                  <Tab icon={<HomeIcon />} label="General" />
                  <Tab icon={<ListIcon />} label="Models" />
                  <Tab icon={<AppsIcon />} label="Dashboard" />
                  
                  
                </Tabs>

                <TabPanel value={view_id} index={0}>
                  {Object.keys(task).map(key => (
                    key!="models" &&
                    
                      <Typography>
                        <Box m={3} component="span" color="primary.main">
                        {key}:
                        </Box>
                        <Box m={1} component="span">
                        {task[key]}
                        </Box>
                      </Typography>
                  ))}
                </TabPanel>
                <TabPanel value={view_id} index={1}>
                  { 
                    task && task.models && view_id==1 && <TableModelsComponent  models={task.models} task_id={task.task_id}/>
                  }
                    
                </TabPanel>
                <TabPanel value={view_id} index={2}>
                  { 
                    task && task.models && view_id==2 && <ModelsDashboard  models={task.models}/>
                  }

                </TabPanel>
                
              
              </DialogContent>
            </Dialog>
          </div>
        );
    }
    
    
    
    handleClose = (event) => {
      this.props.onClose(event)
    };

    handleSubmit = (event) =>{

      
    }
    
    
}

TaskDetailsComponent.propTypes = {
    classes: PropTypes.object.isRequired
};


const TaskDetailsPage = withStyles(styles)(TaskDetailsComponent);

export default TaskDetailsPage