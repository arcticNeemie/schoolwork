import rospy
from geometry_msgs.msg import Twist
import time

def move():
	#Start new node
	rospy.init_node('turtlecontroller',anonymous=True)
	#Find the velocity publisher
	velocity_publisher = rospy.Publisher('/mobile_base/commands/velocity',Twist,queue_size=10)
	velocity_msg = Twist()
	while not rospy.is_shutdown():
		#Forward at 0.3 for 5 secs
		velocity_msg.linear.x = 0.3
		end = time.time() + 5
		while(time.time()<end):
			velocity_publisher.publish(velocity_msg)
			
		velocity_msg.linear.x = 0
		velocity_publisher.publish(velocity_msg)
		#Rotate at 0.1 for 1 sec
		velocity_msg.angular.x = 0.1
		end = time.time() + 1
		while(time.time()<end):
			velocity_publisher.publish(velocity_msg)
			
		velocity_msg.angular.x = 0
		velocity_publisher.publish(velocity_msg)
		
		#Forward at 0.2 for 5 sec
		velocity_msg.linear.x = 0.2
		end = time.time() + 5
		while(time.time()<end):
			velocity_publisher.publish(velocity_msg)
			
		velocity_msg.linear.x = 0
		velocity_publisher.publish(velocity_msg)
		
		#Break
		break
		
if __name__ == '__main__':
	try:
		move()
	except rospy.ROSInterruptException: pass
