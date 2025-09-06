#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Initialize Adafruit PWM Servo Driver
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Servo channels
#define BASE_CHANNEL 0
#define SHOULDER_CHANNEL 1
#define ELBOW_CHANNEL 2
#define GRIPPER_CHANNEL 3

// Pulse width limits (in microseconds)
#define SERVO_MIN 150    // Base, Shoulder, Elbow minimum
#define SERVO_MAX 600    // Base, Shoulder, Elbow maximum
#define GRIPPER_MIN 180  // Gripper minimum
#define GRIPPER_MAX 600  // Gripper maximum

// Angle limits
#define JOINT_MIN 0
#define JOINT_MAX 180
#define GRIPPER_MAX_ANGLE 90

// Adjustable step size and delay
#define STEP_SIZE 5      // Step size in degrees
#define DELAY_MS 50      // Delay between steps (ms)

// Current angles
int currentBase = 90;
int currentShoulder = 70;
int currentElbow = 0;
int currentGrip = 0;

void setup() {
  Serial.begin(9600);
  while (!Serial) { ; } // Wait for serial

  Serial.println("3DOF Arm Servo Control (Automated Mode)");

  pwm.begin();
  pwm.setPWMFreq(60); // Analog servos run at ~60 Hz
  delay(10);

  // Initialize servos to default position
  setServo(BASE_CHANNEL, currentBase);
  setServo(SHOULDER_CHANNEL, currentShoulder);
  setServo(ELBOW_CHANNEL, currentElbow);
  setServo(GRIPPER_CHANNEL, currentGrip);

  Serial.println("Ready to receive commands like: BASE:90 or SHOULDER:120 or GRIP:45");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command.length() > 0) {
      processCommand(command);
    }
  }
}

// Smooth movement function
void moveServoSmooth(uint8_t channel, int &currentAngle, int targetAngle) {
  if (targetAngle == currentAngle) return; // Skip if already at target
  targetAngle = constrain(targetAngle, JOINT_MIN, (channel == GRIPPER_CHANNEL) ? GRIPPER_MAX_ANGLE : JOINT_MAX);

  if (targetAngle > currentAngle) {
    for (int angle = currentAngle; angle <= targetAngle; angle += STEP_SIZE) {
      setServo(channel, angle);
      delay(DELAY_MS);
    }
  } else {
    for (int angle = currentAngle; angle >= targetAngle; angle -= STEP_SIZE) {
      setServo(channel, angle);
      delay(DELAY_MS);
    }
  }
  currentAngle = targetAngle;
}

// Set servo pulse based on angle
void setServo(uint8_t channel, int angle) {
  int pulse;
  if (channel == GRIPPER_CHANNEL) {
    pulse = map(angle, JOINT_MIN, GRIPPER_MAX_ANGLE, GRIPPER_MIN, GRIPPER_MAX);
  } else {
    pulse = map(angle, JOINT_MIN, JOINT_MAX, SERVO_MIN, SERVO_MAX);
  }
  pwm.setPWM(channel, 0, pulse);
}

// Process serial command like BASE:90
void processCommand(String command) {
  int colonIndex = command.indexOf(':');
  if (colonIndex == -1) {
    Serial.println("Error: Invalid command format (missing ':')");
    return;
  }

  String joint = command.substring(0, colonIndex);
  String angleStr = command.substring(colonIndex + 1);
  int angle;

  if (angleStr.length() == 0 || !isNumeric(angleStr)) {
    Serial.println("Error: Invalid angle value");
    return;
  }
  angle = angleStr.toInt();

  if (joint.equalsIgnoreCase("BASE")) {
    if (angle < JOINT_MIN || angle > JOINT_MAX) {
      Serial.println("Error: BASE angle out of range (0-180)");
      return;
    }
    Serial.print("Received BASE:");
    Serial.println(angle);
    moveServoSmooth(BASE_CHANNEL, currentBase, angle);

  } else if (joint.equalsIgnoreCase("SHOULDER")) {
    if (angle < JOINT_MIN || angle > JOINT_MAX) {
      Serial.println("Error: SHOULDER angle out of range (0-180)");
      return;
    }
    Serial.print("Received SHOULDER:");
    Serial.println(angle);
    moveServoSmooth(SHOULDER_CHANNEL, currentShoulder, angle);

  } else if (joint.equalsIgnoreCase("ELBOW")) {
    if (angle < JOINT_MIN || angle > JOINT_MAX) {
      Serial.println("Error: ELBOW angle out of range (0-180)");
      return;
    }
    Serial.print("Received ELBOW:");
    Serial.println(angle);
    moveServoSmooth(ELBOW_CHANNEL, currentElbow, angle);

  } else if (joint.equalsIgnoreCase("GRIP")) {
    if (angle < JOINT_MIN || angle > GRIPPER_MAX_ANGLE) {
      Serial.println("Error: GRIP angle out of range (0-90)");
      return;
    }
    Serial.print("Received GRIP:");
    Serial.println(angle);
    moveServoSmooth(GRIPPER_CHANNEL, currentGrip, angle);

  } else {
    Serial.println("Error: Unknown joint name");
  }
}

// Check if string is numeric
bool isNumeric(String str) {
  for (unsigned int i = 0; i < str.length(); i++) {
    if (!isDigit(str.charAt(i))) return false;
  }
  return true;
}