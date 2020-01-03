import React , { Component } from 'react';

import './App.css';

import { BrowserRouter as Router,
  Switch,
  Route,
  Link } from "react-router-dom"
import DashboardComponent from './pages/dashboard'
import FMComponent from './pages/fm'
import clsx from 'clsx';
import { makeStyles, useTheme, withStyles } from '@material-ui/core/styles';

import IconButton from '@material-ui/core/IconButton';
import MenuIcon from '@material-ui/icons/Menu';
import ChevronLeftIcon from '@material-ui/icons/ChevronLeft';
import ChevronRightIcon from '@material-ui/icons/ChevronRight';

import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemIcon from '@material-ui/core/ListItemIcon';
import ListItemText from '@material-ui/core/ListItemText';

import Typography from '@material-ui/core/Typography';
import Divider from '@material-ui/core/Divider';
import Drawer from '@material-ui/core/Drawer';

import ViewModuleIcon from '@material-ui/icons/ViewModule';
import StorageIcon from '@material-ui/icons/Storage';
import FiberNewIcon from '@material-ui/icons/FiberNew';



const drawerWidth = 140;
const appBarHeight = 70;

const styles = theme => ({
  root: {
    display: 'flex',
  },
  appBar: {
    transition: theme.transitions.create(['margin', 'width'], {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.leavingScreen,
    }),
    height: appBarHeight,
  },
  appBarShift: {
    width: `calc(100% - ${drawerWidth}px)`,
    marginLeft: drawerWidth,
    transition: theme.transitions.create(['margin', 'width'], {
      easing: theme.transitions.easing.easeOut,
      duration: theme.transitions.duration.enteringScreen,
    }),
  },
  menuButton: {
    marginRight: theme.spacing(2),
  },
  hide: {
    display: 'none',
  },
  drawer: {
    width: drawerWidth,
    flexShrink: 0,
  },
  drawerPaper: {
    width: drawerWidth,
  },
  drawerHeader: {
    display: 'flex',
    alignItems: 'center',
    padding: theme.spacing(0, 1),
    ...theme.mixins.toolbar,
    justifyContent: 'flex-end',
  },
  content: {
    flexGrow: 1,
    padding: theme.spacing(3),
    transition: theme.transitions.create('margin', {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.leavingScreen,
    }),
    marginLeft: -drawerWidth,
    marginTop:appBarHeight,
  },
  contentShift: {
    transition: theme.transitions.create('margin', {
      easing: theme.transitions.easing.easeOut,
      duration: theme.transitions.duration.enteringScreen,
    }),
    marginLeft: 0,
  },

  drawerList:{
    alignItems:'center'
  },
});

function RemoteComponent() {
  return <h2>Remote</h2>;
}


class AppComponent extends Component {

  constructor (props) {
    super(props)
    this.state = {
      drawerOpen:false
    }
  }
  

  render() {
    const { classes } = this.props;
    const theme = {"direction":"ltr"}
    const {drawerOpen } = this.state

    const handleDrawerOpen = () => {
      this.setState({ drawerOpen: true });
      
    };
  
    const handleDrawerClose = () => {
      this.setState({ drawerOpen: false });
    };

    return (
      <Router>
        <div className={classes.root}>
          <AppBar position="fixed"
          className={clsx(classes.appBar, {
            [classes.appBarShift]: drawerOpen,
          })}>
            <Toolbar>
              <IconButton
                  color="inherit"
                  aria-label="open drawer"
                  onClick={handleDrawerOpen}
                  edge="start"
                  className={clsx(classes.menuButton, drawerOpen && classes.hide)}
                >
                <MenuIcon />
              </IconButton>

              <Typography variant="h6" color="inherit" className={classes.grow}>
                Dashboard FeatureNet
              </Typography>
            </Toolbar>
          </AppBar>


          <Drawer
            className={classes.drawer}
            variant="persistent"
            anchor="left"
            open={drawerOpen}
            classes={{
              paper: classes.drawerPaper,
            }}
          >
            <div className={classes.drawerHeader}>
              <IconButton onClick={handleDrawerClose}>
                {theme.direction === 'ltr' ? <ChevronLeftIcon /> : <ChevronRightIcon />}
              </IconButton>
            </div>
            <Divider />
            <List>
            
              <ListItem button key={"FM"}>
                <Link to="/fm" style={{ textDecoration: 'none', textAlign:'center', width:"100%"}}>
                  <ListItemIcon><ViewModuleIcon style={{marginLeft:16}} /></ListItemIcon>
                  <ListItemText primary="Feature Model" />
                </Link>
                
              </ListItem>

              <ListItem button key={"CONFIG"}>
                <Link to="/" style={{ textDecoration: 'none' , textAlign:'center', width:"100%"}}>
                  <ListItemIcon><FiberNewIcon style={{marginLeft:16}} /></ListItemIcon>
                  <ListItemText primary="Local" />
                </Link>
              </ListItem>

              <ListItem button key={"DATABASE"}>
                <Link to="/remote" style={{ textDecoration: 'none' , textAlign:'center', width:"100%"}}>
                  <ListItemIcon><StorageIcon style={{marginLeft:16}} /></ListItemIcon>
                  <ListItemText primary="Remote" />
                </Link>
              </ListItem>
          
            </List>

          </Drawer>
          <main
            className={clsx(classes.content, {
              [classes.contentShift]: drawerOpen,
            })}
      >
            <Switch>
              <Route path="/fm">
                <FMComponent />
              </Route>
              <Route path="/">
                <DashboardComponent />
              </Route>
              <Route path="/remote">
                <RemoteComponent />
              </Route>
            </Switch>
          </main> 
        </div>
      </Router>
    );
  }
}
export default withStyles(styles)(AppComponent);


