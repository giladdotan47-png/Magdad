// MultiStepper.pde
// -*- mode: C++ -*-
//
// Shows how to multiple simultaneous steppers
// Runs one stepper forwards and backwards, accelerating and decelerating
// at the limits. Runs other steppers at the same time
//
// Copyright (C) 2009 Mike McCauley
// $Id: MultiStepper.pde,v 1.1 2011/01/05 01:51:01 mikem Exp mikem $

#include <AccelStepper.h>

const int StepX = 5;
const int DirX = 2;
const int StepY = 6;
const int DirY = 3;
const int StepZ = 7;
const int DirZ = 4;
const float speed = 300.0;


// Define some steppers and the pins the will use
AccelStepper stepper1(AccelStepper::DRIVER, StepX, DirX);
AccelStepper stepper2(AccelStepper::DRIVER, StepY, DirY);
AccelStepper stepper3(AccelStepper::DRIVER, StepZ, DirZ);

void setup()
{  
    stepper1.setMaxSpeed(speed);
    stepper1.setAcceleration(speed);
    stepper1.moveTo(200);
    
    stepper2.setMaxSpeed(speed);
    stepper2.setAcceleration(speed);
    stepper2.moveTo(1000000);
    
    stepper3.setMaxSpeed(speed);
    stepper3.setAcceleration(speed);
    stepper3.moveTo(1000000); 
}



void loop()
{
    // Change direction at the limits
    stepper1.run();
    stepper2.run();
    stepper3.run();
}
