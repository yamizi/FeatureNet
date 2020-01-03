import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';


import Fab from '@material-ui/core/Fab';
import AddIcon from '@material-ui/icons/Add';
import DeleteIcon from '@material-ui/icons/Delete';

import PipelineHomePage from './pipelineHome'
import PipelineTablePage from './taskTable'
import TaskDetailsPage from './taskDetails'

import API from '../api';

const user =  {
  avatar:"http://www.techschool.lu/images/Logo_Small.png",
  firstName:"Salah",
  lastName:"Ghamizi",
  right:"user",
  id:0
}
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

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI.
    return { hasError: true, newTask:false ,selectedTask:false };
  }

  componentDidCatch(error, errorInfo) {
    // You can also log the error to an error reporting service
    console.log(error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      // You can render any custom fallback UI
      return <PipelineTablePage usr={user} onTaskSelect={this.handleSelectTaskClickOpen}/>;
    }

    return this.props.children; 
  }
}

class DashboardComponent extends React.Component {

  constructor(props) {
    // Required step: always call the parent class' constructor
    super(props);

    let date= new Date();

    // Set the state directly. Use props if necessary.
    this.state = {

      user: user,
      newTask: false,
      selectedTask: false
    }

  }

  delete_tasks = (event) =>{
    API.delete(`sample/`, {userId:this.state.user.id})
      .then(res => {
          if(res.data){
                
        }
        
      })
  }


  closeDetailsTask = (event) =>{
    this.setState({ selectedTask: null });
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

  handleSelectTaskClickOpen = (task) => {
    this.setState({ selectedTask: task });
  };

  render() {
    const { classes } = this.props;
    const {user, newTask,selectedTask } = this.state
    return (
      <div className={classes.root}>        

        <Fab color="primary" aria-label="Add" className={classes.fab} onClick={this.handleNewTaskClickOpen}>
          <AddIcon />
        </Fab>

        <Fab aria-label="Delete all" className={classes.fabDelete} onClick={this.delete_tasks}>
          <DeleteIcon  />
        </Fab>

        { 
          newTask && <PipelineHomePage onClose={this.closeNewTask}/>
        }

        { 
          selectedTask && <TaskDetailsPage usr={user} task={selectedTask} onClose={this.closeDetailsTask}/>
        }

        { 
        
          !newTask && !selectedTask && 
          <ErrorBoundary>
            <PipelineTablePage usr={user} onTaskSelect={this.handleSelectTaskClickOpen}/>
          </ErrorBoundary>

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