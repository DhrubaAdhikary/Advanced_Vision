import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GLU import *
from OpenGL.GL import *

class Camera:
    # This class is unchanged from the last version (with top-down view and zoom)
    def __init__(self, fov=45, cam_distance_=100) -> None:
        self.position = (0.0, 0.0, 0.0)
        self.orbital_radius = cam_distance_
        self.polar = np.deg2rad(89.9); self.azimuth = np.deg2rad(-90)
        self.setup(fov_y=fov, aspect_ratio=800/600, near=1.0, far=50000.0)

    def setup(self, fov_y, aspect_ratio, near, far):
        glMatrixMode(GL_PROJECTION); glLoadIdentity(); gluPerspective(fov_y, aspect_ratio, near, far); glMatrixMode(GL_MODELVIEW); glLoadIdentity()
    def rotate_azimuth(self, degree=0.0): self.azimuth += np.deg2rad(degree)
    def rotate_polar(self, degree=0.0):
        self.polar += np.deg2rad(degree); cap = np.deg2rad(89.99); self.polar = max(min(self.polar, cap), -cap)
    def zoom(self, by=0.0): self.orbital_radius = max(1, self.orbital_radius + by)
    def update(self, rotation_center):
        x = self.orbital_radius * np.cos(self.polar) * np.cos(self.azimuth) + rotation_center[0]
        y = self.orbital_radius * np.sin(self.polar) + rotation_center[1]
        z = self.orbital_radius * np.cos(self.polar) * np.sin(self.azimuth) + rotation_center[2]
        glMatrixMode(GL_MODELVIEW); glLoadIdentity(); gluLookAt(x, y, z, *rotation_center, 0, 1, 0)

class Renderer3D:
    def __init__(self, pov_=90.0, cam_distance=100) -> None:
        pygame.init()
        self.window = (800, 600)
        pygame.display.set_mode(self.window, DOUBLEBUF | OPENGL)
        pygame.display.set_caption('3D SLAM Visualization')
        self.camera = Camera(fov=pov_, cam_distance_=cam_distance)
        glEnable(GL_DEPTH_TEST); glPointSize(2)
        self.view_center = np.array([0.0, 0.0, 0.0]); self.is_panning_manual = False

    def handle_inputs(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); quit()
            if event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[2]:
                self.camera.rotate_azimuth(-event.rel[0] * 0.5)
                self.camera.rotate_polar(event.rel[1] * 0.5)
            if event.type == pygame.MOUSEWHEEL: self.camera.zoom(event.y * -10)
        keys = pygame.key.get_pressed(); pan_speed = 0.02 * self.camera.orbital_radius; panned = False
        if keys[pygame.K_w]: self.view_center[2] -= pan_speed; panned = True
        if keys[pygame.K_s]: self.view_center[2] += pan_speed; panned = True
        if keys[pygame.K_a]: self.view_center[0] -= pan_speed; panned = True
        if keys[pygame.K_d]: self.view_center[0] += pan_speed; panned = True
        if keys[pygame.K_r]: self.is_panning_manual = False
        if panned: self.is_panning_manual = True
        if keys[pygame.K_ESCAPE]: pygame.quit(); quit()

    def draw_lines(self, start, end, color):
        glColor3f(*color); glBegin(GL_LINES); glVertex3f(*start); glVertex3f(*end); glEnd()

    def render_axis(self):
        axis_len = self.camera.orbital_radius * 0.1
        self.draw_lines((0,0,0), (axis_len, 0, 0), (1,0,0)); self.draw_lines((0,0,0), (0, axis_len, 0), (0,1,0)); self.draw_lines((0,0,0), (0, 0, axis_len), (0,0,1))

    def draw_trajectory(self, camera_poses):
        glColor3f(1.0, 0.5, 0.0); glBegin(GL_LINE_STRIP)
        for p in camera_poses: glVertex3fv(p['t'])
        glEnd()
        glColor3f(1.0, 0.0, 0.0)
        for p in camera_poses:
            glPushMatrix(); glTranslatef(*p['t']); cube_size = self.camera.orbital_radius * 0.01; glScalef(cube_size, cube_size, cube_size)
            vertices=[(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,1,1),(-1,1,1),(-1,1,1)]; edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)];
            glBegin(GL_LINES);
            for e in edges: glVertex3fv(vertices[e[0]]); glVertex3fv(vertices[e[1]]);
            glEnd(); glPopMatrix()

    def draw_points(self, points):
        """A simple method that just draws points it's given."""
        glBegin(GL_POINTS); glColor3f(0.0, 1.0, 1.0)
        for p in points:
            glVertex3f(p[0], p[1], p[2]) # Draw the point at its GL coordinate
        glEnd()

    def render3dSpace(self, points_to_render, trajectory_poses):
        glClearColor(0.1, 0.1, 0.1, 1.0); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.handle_inputs()
        if not self.is_panning_manual and trajectory_poses: self.view_center = trajectory_poses[-1]['t']
        self.camera.update(rotation_center=self.view_center)
        self.render_axis()
        if trajectory_poses: self.draw_trajectory(trajectory_poses)
        
        # This function now receives world points ready for OpenGL
        # and simply draws them.
        if points_to_render is not None and len(points_to_render) > 0:
            self.draw_points(points_to_render)
        
        pygame.display.flip()