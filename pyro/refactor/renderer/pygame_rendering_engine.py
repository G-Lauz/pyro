from typing import Tuple

from pyro.refactor.renderer.base import RenderingEngine


class PygameRenderingEngine(RenderingEngine):
    def __init__(self, caption: str = "Pygame Renderer"):
        super().__init__()

        # Lazy import pygame
        import pygame
        self.pygame = pygame

        self.pygame.init()
        self.pygame.font.init()

        self.screen_size = (800, 600)  # Default screen size
        self.screen = None
        self.font = None

        self.caption = caption

    def init(self, screen_size=None):
        if screen_size is not None:
            self.screen_size = screen_size

        self.screen = self.pygame.display.set_mode(self.screen_size)
        self.pygame.display.set_caption(self.caption)
        self.font = self.pygame.font.SysFont("Arial", 20)

    def clear_screen(self, color:Tuple[int, int, int] = (0, 0, 0)):
        self.screen.fill(color)

    def draw_polygon(self, points, color, width=0):
        self.pygame.draw.polygon(self.screen, color, points, width)

    def draw_line(self, start, end, color, width=1):
        self.pygame.draw.line(self.screen, color, start, end, width)

    def draw_lines(self, points, color, width=1):
        self.pygame.draw.lines(self.screen, color, False, points, width)

    def draw_text(self, text, position, color):
        text_surface = self.font.render(text, True, color)
        self.screen.blit(text_surface, position)

    def update_display(self):
        self.pygame.display.flip()
