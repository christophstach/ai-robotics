#!/usr/bin/env bash

clear
catkin_make
source ./install/setup.zsh
pipenv shell
rosrun excercise_1 Prediction.py