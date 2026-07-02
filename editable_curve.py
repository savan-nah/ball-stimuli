#!/usr/bin/env python
# -*- coding: utf-8 -*-
#From: C++ version Copyright (c) 2006-2007 Erin Catto http://www.box2d.org
# Python version Copyright (c) 2010 kne / sirkne at gmail dot com
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
# 1. The origin of this software must not be misrepresented; you must not
# claim that you wrote the original software. If you use this software
# in a product, an acknowledgment in the product documentation would be
# appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
# misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.

"""
Savannah Parsons's attempt to create a simple, self-contained pygame-based stimulus that features: 

Four static bodies:
    + First fixture:  big polygon to represent the ground
    + Second fixture: big polygon to represent the right wall
    + Third fixture:  big polygon to represent the left wall
    + Fourth fixture: big polygon to represent the top

Two dynamic bodies:
    + First fixture: a "leader" circle whose position is tracked
    + Second fixture: a "follower" circle that is given the leader position, and follows its path along the screen
"""
import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
from collections import deque #added to allow for delayed steps from follower - ClaudeAI suggestion
import cv2          #to save video
import numpy as np  #to save video
import os
import math #to prevent corner bouncing
import random

import Box2D  # The main library
from Box2D import (b2Vec2, b2Filter, b2ContactListener)
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody, kinematicBody)

# --- constants ---
# Box2D deals with meters, but we want to display pixels,
# so define a conversion factor:
PPM = 20.0        # pixels per meter
TARGET_FPS = 60
PHYSICS_STEPS = 3 #(180Hz physics)
TIME_STEP = 1.0 / TARGET_FPS / PHYSICS_STEPS
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480

radius = 1.0
gap = 0.5

LAG_METERS = 4                # meters along path behind leader (tune: 2.5=tight, 4=looser)
MIN_RECORD_DIST = 0.01

# --- video --- #
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, TARGET_FPS, (SCREEN_WIDTH, SCREEN_HEIGHT))

#Saving video 
frames = []
MAX_FRAMES = TARGET_FPS * 20  # 150 fps * 20 seconds = 3000 frames
frame_count = 0

# --- pygame setup --
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Linear Movement Render Draft')
clock = pygame.time.Clock()

# --- Wall setup ---#
world = world(gravity=(0, 0), doSleep=True) #0 gravity allows shapes to move around the plac

# Add the static body to hold the ground shape
ground_body = world.CreateStaticBody(position=(0,0))  #pybox2d expects HALF-EXTENTS. This is really 32, 0 and accounts for the ball's radius to ensure it's not being sent off screen
ground_body.CreateFixture(
    shape=polygonShape(box=(16, 0.5)),                  #pybox2d expects HALF-EXTENTS. This is really 32, 1
    friction=0.0,
    restitution = 1.0
)

# Adding a static body to hold the right wall shape
right_wall_body = world.CreateStaticBody(position=(32.5 - (radius + gap), 12)) #x and y coordinates on screen 
right_wall_body.CreateFixture(
    shape=polygonShape(box=(0.5, 12)),                      #size of box - in this case, 1m wide and 24m tall - IN HALF EXTENTS
    friction=0.0,
    restitution = 1.0
)

#And the left
left_wall_body = world.CreateStaticBody(position=(-0.5 + (radius + gap), 12))
left_wall_body.CreateFixture(
    shape=polygonShape(box=(0.5, 12)),
    friction=0.0,
    restitution = 1.0
)

#And top
top_wall_body = world.CreateStaticBody(position=(16, 24.5 - (radius + gap)))
top_wall_body.CreateFixture(
    shape=polygonShape(box=(16, 0.5)),
    friction=0.0,
    restitution = 1.0
)

# --- STARTING POSITIONS FOR LEADER AND FOLLOWER --- #
leader_start_x = 1.25
leader_start_y = 12.0 

follower_start_y = leader_start_y - (2 * radius + gap)

#Make the bodies! 
leader = world.CreateDynamicBody(position=(leader_start_x, leader_start_y))     #leader is a kinematic, physical body that moves according to our movement_gradient function
leader.CreateCircleFixture(
    radius=radius,                    
    density=1,
    restitution=1.0,          #restitution allows for "bounciness"
    friction=0.0
)

#Give leader velocity 
# leader.linearVelocity = (18, 7)

#And its position history is recorded - Cursor
position_history = deque()

def record_leader_position(history, leader_pos):
    """This code ensures that, during the rendering, the leader position is appended only when it has moved 
    enough to matter"""
    if not history:
        history.append(b2Vec2(leader_pos.x, leader_pos.y))
        return
    
    last = history[-1]
    dx = leader_pos.x - last.x
    dy = leader_pos.y - last.y
    if (dx * dx + dy * dy) ** 0.5 >= MIN_RECORD_DIST:
        history.append(b2Vec2(leader_pos.x, leader_pos.y))

# Recording the path length - Cursor 
def path_length(history):
    """Sum of segment lengths along the recorded polyline."""
    total = 0.0
    for i in range(1, len(history)):
        total += (history[i] - history[i - 1]).length
    return total

#Ensuring that the follower always tracks along the path, not just some distance behind - Cursor
def position_at_arc_length(history, target_s):
    """
    Walk along history[0] -> history[1] -> ... until we've travelled target_s meters.
    Return (x, y) at that point (may interpolate within a segment).
    """
    if not history:
        return None
    if target_s <= 0:
        return b2Vec2(history[0].x, history[0].y)

    travelled = 0.0
    for i in range(1, len(history)):
        seg = history[i] - history[i - 1]
        seg_len = seg.length
        if seg_len == 0:
            continue

        if travelled + seg_len >= target_s:
            # target lies inside this segment — interpolate
            t = (target_s - travelled) / seg_len
            x = history[i - 1].x + seg.x * t
            y = history[i - 1].y + seg.y * t
            return b2Vec2(x, y)

        travelled += seg_len

    # Path isn't long enough yet — return oldest point (follower waits at start)
    return b2Vec2(history[0].x, history[0].y)

def follower_position_arc_lag(history, lag_meters):
    """Follower sits lag_meters behind the leader along the recorded path."""
    if not history:
        return None
    if len(history) < 2:
        return b2Vec2(history[0].x, history[0].y)

    total = path_length(history)
    target_s = total - lag_meters

    if target_s < 0:
        return b2Vec2(history[0].x, history[0].y)

    return position_at_arc_length(history, target_s)

#Pre-fill so follower starts on-path behind leader (not off-screen)
for i in range(LAG_METERS):
    position_history.append(b2Vec2(leader_start_x, follower_start_y))

colors = {
    staticBody: (255, 255, 255, 255)
}

def my_draw_polygon(polygon, body, fixture):
#interp. This code draws ALL the polygons that define our walls/top and bottom 
    vertices = [(body.transform * v) * PPM for v in polygon.vertices]
    vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
    pygame.draw.polygon(screen, colors[body.type], vertices)
polygonShape.draw = my_draw_polygon

def my_draw_circle(circle, body, fixture):
#interp. This code draws the leader stimuli, as we later manually draw the follower
    position = body.transform * circle.pos * PPM
    position = (position[0], SCREEN_HEIGHT - position[1])
    color = (0, 255, 0)
    pygame.draw.circle(screen, color, [int(
        x) for x in position], int(circle.radius * PPM))
    # Note: Python 3.x will enforce that pygame get the integers it requests,
    #       and it will not convert from float.
circleShape.draw = my_draw_circle

#---FOR EDITABLE PARAMETRIC CURVE ---#

# TODO - make it stop COLLIDING!!!! 

# -MAKE PAUSE- #

class CollisionDetector(b2ContactListener): #the contact listener does not actually DO something upon contact
    """
    This code creates a Collision_Detector that uses the existing b2ContactListener from pybox2d as a
    parent class, and then changes some of the attributes to allow the leader to "pause" on wall contact
    """
    def __init__(self): #init = create a new instance of the CollisionDetector class
        super().__init__() #before doing what I need you to do, run everything that Contact Listener does to 
        #avoid breaking something 
        self.hit = False              #no collision has occurred
    def BeginContact(self, contact):  #if two objects contact, return true for my pause function
        self.hit = True

listener = CollisionDetector()
world.contactListener = listener

def pause(time_length: float, jitter: float = 0.0) -> None:
    """
    This code allows the researcher to edit the amount of time a ball pauses once a collision is detected.
    Jitter: allows the follower to shake while it "waits", similar to Heider simmel movies. In pixels. 
    0 = it does not shake
    delay: jitter only after the first actual bounce, not at startup
    Claude-supported (full Claude = the jitter)
    """
    if pause.first_hit: 
        pause.first_hit = False 
        listener.hit = False
        return 

    this_jitter = random.uniform(jitter * 0.2, jitter)

    start_time = pygame.time.get_ticks()
    end_time = pygame.time.get_ticks() + int(time_length * 1000)

    impact_x = leader.position.x * PPM
    impact_y = SCREEN_HEIGHT - (leader.position.y * PPM)

    history_snapshot = list(position_history)
    total_steps      = len(history_snapshot) - 10

    #MAX ADVANCE - so follower doesn't overlap with leader
    MAX_ADVANCE = 0.3

    while pygame.time.get_ticks() < end_time:
        # keep pygame responsive during pause
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                return
    
        elapsed  = (pygame.time.get_ticks() - start_time) / 1000.0

        # draw leader with a sway
        sway_leader_x = impact_x + this_jitter * math.sin(elapsed * math.pi * 2)
        sway_leader_y = impact_y + (this_jitter*0.2) * math.sin(elapsed * math.pi * 1)

        #and also the follower!
        old_pos = follower_position_arc_lag(position_history, LAG_METERS)
        follower_base_x = old_pos.x * PPM
        follower_base_y = SCREEN_HEIGHT - (old_pos.y * PPM)
        
        sway_follower_x = follower_base_x + this_jitter * math.sin(elapsed * math.pi * 0.6)
        sway_follower_y = follower_base_y + (this_jitter * 0.2) * math.sin(elapsed * math.pi * 0.6)

        screen.fill((255, 255, 255, 255))

        for body in world.bodies:
            if body == leader:
                continue
            for fixture in body.fixtures:
                fixture.shape.draw(body, fixture)

        pygame.draw.circle(screen, (0, 255, 0),
                            (int(sway_leader_x), int(sway_leader_y)),
                            int(radius * PPM))

        pygame.draw.circle(screen, (0, 255, 0),
                            (int(sway_follower_x), int(sway_follower_y)),
                            int(radius * PPM))

        # jitter_x = old_pos.x * PPM + random.uniform(-jitter, jitter)
        # jitter_y = SCREEN_HEIGHT - (old_pos.y * PPM) + random.uniform(-jitter, jitter)
        # pygame.draw.circle(screen, (0, 255, 0),
        #                    (int(jitter_x), int(jitter_y)),
        #                    int(radius * PPM))
        
  
         

        # # draw follower with sway
        # old_pos = position_history[0]
        # sway_x = old_pos.x * PPM + offset
        # sway_y = SCREEN_HEIGHT - (old_pos.y * PPM)
        # pygame.draw.circle(screen, (0, 255, 0),
        #                    (int(sway_x), int(sway_y)),
        #                    int(radius * PPM))
        
             # how far through the pause are we (0.0 → 1.0)
    #  elapsed = (pygame.time.get_ticks() - start_time) / 1000.0

        pygame.display.flip()
        clock.tick(TARGET_FPS)

pause.first_hit = True 

# # - MAKE LINEAR DELINEATION -#

#OG - it works! 
# def get_next_target(pos):
#     """
#     Aims at the wall diagonally opposite from the leader's current position.
#     "Diagonal" here is determined by how far the x-coordinate versus the y-coordinate is from centre. If the y-coordinate is further
#     from centre point 12 than x is from 16, the "diagonal" wall is either top or bottom. If the x-coordinate is further from 16 than y
#     is from 24, "opposite" is left/right wall 
#     """
#     horizontal_deviation = abs(pos[0] - 16)
#     vertical_deviation = abs(pos[1] - 12)

#     right_wall = pos[0] > 16
#     top_wall = pos[1] > 12

#     if horizontal_deviation > vertical_deviation:
#         if right_wall:
#             target_x = 1.25
#             target_y = random.uniform(1.5, 22.5)  
#         else:
#             target_x = 30.75
#             target_y = random.uniform(1.5, 22.5) 


#     elif horizontal_deviation == vertical_deviation: 
#         if right_wall:
#             target_x = 1.25
#             target_y = random.uniform(1.5, 22.5)
#         else:
#             target_x = 30.75
#             target_y = random.uniform(1.5, 22.5) 


#     elif vertical_deviation > horizontal_deviation:
#         if top_wall:
#             target_x = random.uniform(1.25, 30.75)
#             target_y = 1.5
#         else: 
#             target_x = random.uniform(1.25, 30.75)
#             target_y = 22.5

#     return (target_x, target_y)

def get_next_target(pos):
    """
    Aims at the wall diagonally opposite from the leader's current position.
    Rather than diagonal using the right_wall/top_wall fixed points, the ball now "randomly chooses" an axis as the next target position
    """

    right_wall = pos[0] > 16
    top_wall = pos[1] > 12

    diagonal_choice = random.choice(["vertical axis", "horizontal axis"])

    if diagonal_choice == "horizontal axis":
        if right_wall: 
            target_x = 0.5
            target_y = random.uniform(0.5, 23.5)
        else:
            target_x = 31.5
            target_y = random.uniform(0.5, 23.5) 

    elif diagonal_choice == "vertical axis":
        if top_wall:
            target_x = random.uniform(0.5, 31.5)
            target_y = 0.5
        else: 
            target_x = random.uniform(0.5, 31.5)
            target_y = 23.5

    return (target_x, target_y)
        

def linear_delineation(P1_offset_x, P1_offset_y): #P1 is representative of the middle point in a parametric curve, or the peak
    """
    This parametrizes a curve between three points (ClaudeAI supported):

    P0: the current Box2D position of the leader
    P2: Computes where ball should land as the diagonally "opposite" wall 
    P1: the midpoint, shifted by P1 offset

    For linear: P1 should be (0,0)
    For parabolic: P1 should be > (0, 0). The larger the values, the more pronounced the curve

    Returns a function r(t) that gives (x,y) along the curve
    """
    P0 = (leader.position.x, leader.position.y)
    P2 = get_next_target(P0)

    P1 = [
        (P0[0] + P2[0]) / 2 + P1_offset_x,       #x-coordinate
        (P0[1] + P2[1]) / 2 + P1_offset_y        #y-coordinate
    ]
    
    def r(t):
        x = (1-t)**2 * P0[0] + 2*(1-t)*t * P1[0] + t**2 * P2[0]
        y = (1-t)**2 * P0[1] + 2*(1-t)*t * P1[1] + t**2 * P2[1]
        return (x, y)
    
    return r

def movement_gradient(P1_offset_x, P1_offset_y, t_speed, pause_time, jitter) -> None:
    """
    This function takes two parameters: 
    curve_amount: Records the level of linear delineation - it is 0, or "no" gravitational pull for a straight line
    and < 0 for a curve downwards 
    t_speed: how fast the ball moves along the parametrized curve
    pause_time: from pause function, makes the leader ball pause for a certain amount of time after wall collision
    """
    if listener.hit:
        if pause_time > 0.0:
            pause(pause_time, jitter)
        listener.hit = False

    t = movement_gradient.t
    if t >= 1.0:
        movement_gradient.t = 0.0 # reset to 0
        # reached P2 — create a new curve from current position
        movement_gradient.current_curve = linear_delineation(P1_offset_x, P1_offset_y)
        leader.position = movement_gradient.current_curve(movement_gradient.t)
    else:
        movement_gradient.t = t + t_speed
        if movement_gradient.t >= 1.0:
             movement_gradient.t = 1.0 # clamp to avoid it going past 1
        leader.position = movement_gradient.current_curve(movement_gradient.t)
        
movement_gradient.current_curve = linear_delineation(0, 0)
movement_gradient.t = 0.0

# --- main game loop ---
pygame.init()

prev_follower_pos = b2Vec2(leader_start_x, follower_start_y - (2 * radius + gap))
record_leader_position(position_history, leader.position)

running = True
while running:
    # Check the event queue
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            # The user closed the window or pressed escape
            running = False

    for i in range(PHYSICS_STEPS):
        world.Step(TIME_STEP, 100, 40)

    #TODO edit this as needed for linear v parabolic
    # movement_gradient(P1_offset = (0,0), pause_time=0.0, ) #this is linear
    # movement_gradient(0, 0, 0) #this is linear - no curve, pause, or jitter
    # movement_gradient(-25, 2.0, 12.0) #this is parabolic with pause, and makes the leader "sway"
    # movement_gradient(5, 0, 0.5, 0.0, 0.0) #big parabolic, pause no jitter
    movement_gradient(10, 10, 0.009, 0.75, 0.0) 

    record_leader_position(position_history, leader.position)
    follower_draw_pos = follower_position_arc_lag(position_history, LAG_METERS)
    if follower_draw_pos is not None:
        prev_follower_pos = b2Vec2(follower_draw_pos.x, follower_draw_pos.y)

    # # 1. Record where the leader is currently, but ensure the follower only moves when the leader is a minimum
    # #distance away, to prevent overlap
    # position_history.append(leader.position.copy())

    # MIN_DIST = 3.0  # minimum separation in meters (> 2 * radius of 1.0)
    # MAX_STEP = 0.3
    
    # follower_draw_pos = position_history[0].copy() 

    # valid_found = False

    # leader_velocity = leader.position - position_history[-2] if len(position_history) > 1 else b2Vec2(0,0)

    # forward = b2Vec2(0,0) 

    # if leader_velocity.length > 0:
    #     forward = leader_velocity.copy()
    #     forward.Normalize()

    # #check if the follower is in front of the leader in its motion direction 
    # prev_follower_pos = follower_draw_pos.copy()

    # best_pos = None
    # best_dist = float('inf')

    # for pos in reversed(position_history): 
    #     offset = pos - leader.position
    #     dist = offset.length

    #     if dist < MIN_DIST:
    #         continue

    #     if leader_velocity.length >0 and dist > 0:
    #             offset_dir = offset.copy()
    #             offset_dir.Normalize()
    #             if offset_dir.dot(forward) > 0.5:
    #                 continue
        
    #     move_dist = (pos - prev_follower_pos).length

    #     if move_dist < best_dist:
    #         best_dist = move_dist
    #         best_pos = pos

    # if best_pos is not None:
    #     move_vec = best_pos - prev_follower_pos

    #     if move_vec.length > MAX_STEP:
    #         move_vec.Normalize()
    #         follower_draw_pos = prev_follower_pos + move_vec * MAX_STEP
    #     else:
    #         follower_draw_pos = best_pos
    
    # if not valid_found:
    #     if leader_velocity.length > 0: 
    #         follower_draw_pos = leader.position - forward * MIN_DIST

#Overlap check:
    # if dist < MIN_DIST and dist > 0:
    #     follower_draw_pos = leader.position + offset.Normalize() * MIN_DIST

    #--- DRAWING --- #

    screen.fill((255, 255, 255, 255))

    # Draw the world
    for body in world.bodies:
        for fixture in body.fixtures:
            fixture.shape.draw(body, fixture)

    #Draw the kinematic follower 
    old_pos = follower_draw_pos
    follower_screen_x = old_pos.x * PPM
    follower_screen_y = SCREEN_HEIGHT - (old_pos.y * PPM)
    pygame.draw.circle(screen, (0, 255, 0), 
                   (int(follower_screen_x), int(follower_screen_y)), 
                   int(1.0 * PPM))
    
    # Capture frames for video
    if frame_count < MAX_FRAMES:
        frame = pygame.surfarray.array3d(screen)
        frame = np.transpose(frame, (1, 0, 2))
        frames.append(frame)
        frame_count += 1
    else:
        running = False  # stop the loop after 20 seconds

    # Flip the screen and try to keep at the target FPS
    pygame.display.flip()
    clock.tick(TARGET_FPS)

pygame.quit()

def get_version(base, ext): 
    #this code counts the incrementing versions of the saved video
    version = 1 #acc, type int
    while os.path.exists(f'{base}{version}{ext}'):
        version += 1
    return f'{base}{version}{ext}'

print('Saving video...')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
filename = get_version('editable_curve_pause', '.mp4')
out = cv2.VideoWriter(filename, fourcc, TARGET_FPS, (SCREEN_WIDTH, SCREEN_HEIGHT))
for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
out.release()

print('Done!')
