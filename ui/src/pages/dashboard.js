import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import Button from '@material-ui/core/Button';
import IconButton from '@material-ui/core/IconButton';
import MenuIcon from '@material-ui/icons/Menu';
import Avatar from '@material-ui/core/Avatar';

import Fab from '@material-ui/core/Fab';
import AddIcon from '@material-ui/icons/Add';
import DeleteIcon from '@material-ui/icons/Delete';

import PipelineHomeComponent from './pipelineHome'
import PipelineTableComponent from './taskTable'

import API from '../api';

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
    position: 'fixed',
    bottom: theme.spacing.unit * 2,
    right: theme.spacing.unit * 2,
  },
  fabDelete: {
    margin: theme.spacing.unit,
    position: 'fixed',
    bottom: theme.spacing.unit * 2,
    right: theme.spacing.unit * 10,
  },
  
});

class DashboardComponent extends React.Component {

  constructor(props) {
    // Required step: always call the parent class' constructor
    super(props);

    let date= new Date();

    // Set the state directly. Use props if necessary.
    this.state = {

      user: {
        avatar:"http://www.techschool.lu/images/Logo_Small.png",
        firstName:"Salah",
        lastName:"Ghamizi",
        right:"user",
        id:0
      },
      newTask: false
    }

  }

  delete_tasks = (event) =>{
    API.delete(`sample/`, {userId:this.state.user.id})
      .then(res => {
          if(res.data){
                
        }
        
    })
}

  closeNewTask = (event) =>{
    this.setState({ newTask: false });
  }

  closeSelectedTask = (event) =>{
    this.setState({ selectedTask: null });
  }

  handleNewTaskClickOpen = () => {
    this.setState({ newTask: true });
  };

  handleSelectTaskClickOpen = (campaign) => {
    this.setState({ selectedTask: campaign });
  };

  render() {
    const { classes } = this.props;
    const {user, newTask } = this.state
    return (
      <div className={classes.root}>
        <AppBar position="static">
          <Toolbar>
            <Avatar alt="User" src={user.avatar} className={classes.avatar} />
            <Typography variant="h6" color="inherit" className={classes.grow}>
              Dashboard FeatureNet
            </Typography>
          </Toolbar>
        </AppBar>

        

        <Fab color="primary" aria-label="Add" className={classes.fab} onClick={this.handleNewTaskClickOpen}>
          <AddIcon />
        </Fab>

        <Fab aria-label="Delete all" className={classes.fabDelete} onClick={this.delete_tasks}>
          <DeleteIcon  />
        </Fab>

        { 
          newTask && <PipelineHomeComponent onClose={this.closeNewTask}/>
        }

        { 
          !newTask && <PipelineTableComponent usr={user}/>
        }

        
      </div>
    );
  }
}

DashboardComponent.propTypes = {
  classes: PropTypes.object.isRequired,
};


const DashboardPage = withStyles(styles)(DashboardComponent);

export default DashboardPage