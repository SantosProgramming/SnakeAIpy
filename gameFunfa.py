import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 100
EXPLORE_PARAM = math.sqrt(2)

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.score = 0
        self.visits = 0

    def is_fully_expanded(self):
        return len(self.children) == len(Direction)

    def select_child(self):
        log_total_visits = math.log(self.visits)

        def ucb_score(child):
            exploration = math.sqrt(log_total_visits / child.visits)
            return child.score / child.visits + EXPLORE_PARAM * exploration

        return max(self.children, key=ucb_score)

    def expand(self):
        unexpanded_directions = [d for d in Direction if not self.has_child_with_direction(d)]
        selected_direction = random.choice(unexpanded_directions)
        new_state = self.state.next_state(selected_direction)
        new_child = Node(new_state, parent=self)
        self.children.append(new_child)
        return new_child

    def has_child_with_direction(self, direction):
        return any(child.state.direction == direction for child in self.children)

    def update_score(self, score):
        self.score += score
        self.visits += 1

    def backpropagate(self, score):
        self.update_score(score)
        if self.parent:
            self.parent.backpropagate(score)


class GameState:
    def __init__(self, game):
        self.game = game
        self.direction = game.direction
        self.snake = game.snake[:]
        self.food = game.food

    def next_state(self, direction):
        game_copy = SnakeGameAI()
        game_copy.direction = direction
        game_copy.head = self.snake[0]
        game_copy.snake = self.snake[:]
        game_copy.food = self.food
        game_copy.play_step()
        return GameState(game_copy)


class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head, Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action=None):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        if action:
            self._move(action)
        else:
            self._move(self._get_next_direction())

        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        # Ensure the new direction is not the opposite direction (180-degree turn)
        if self.direction == Direction.RIGHT and new_dir == Direction.LEFT:
            new_dir = self.direction
        elif self.direction == Direction.LEFT and new_dir == Direction.RIGHT:
            new_dir = self.direction
        elif self.direction == Direction.UP and new_dir == Direction.DOWN:
            new_dir = self.direction
        elif self.direction == Direction.DOWN and new_dir == Direction.UP:
            new_dir = self.direction

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        new_head = Point(x, y)

        if self.is_collision(new_head) or not self._is_within_boundaries(new_head) or new_head in self.snake[1:]:
            # Find a valid path using BFS
            path = self._find_path(new_head)
            if path:
                self.direction = path[0]
                self.head = new_head
            else:
                # If no valid path found, choose a different direction randomly
                directions = [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP]
                directions.remove(self.direction)
                random.shuffle(directions)
                for direction in directions:
                    x = self.head.x
                    y = self.head.y
                    if direction == Direction.RIGHT:
                        x += BLOCK_SIZE
                    elif direction == Direction.LEFT:
                        x -= BLOCK_SIZE
                    elif direction == Direction.DOWN:
                        y += BLOCK_SIZE
                    elif direction == Direction.UP:
                        y -= BLOCK_SIZE

                    new_head = Point(x, y)
                    if not self.is_collision(new_head) and self._is_within_boundaries(new_head) and new_head not in self.snake[1:]:
                        self.direction = direction
                        self.head = new_head
                        return

        if self.is_collision(new_head) or not self._is_within_boundaries(new_head):
            return

        self.head = new_head

    def _find_path(self, target):
        visited = set()
        queue = [[(self.head, self.direction)]]

        while queue:
            path = queue.pop(0)
            current_pos, current_dir = path[-1]

            if current_pos == target:
                return [dir for (_, dir) in path[1:]]

            if current_pos in visited:
                continue

            visited.add(current_pos)

            for direction in [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP]:
                if direction == current_dir:
                    continue

                x = current_pos.x
                y = current_pos.y

                if direction == Direction.RIGHT:
                    x += BLOCK_SIZE
                elif direction == Direction.LEFT:
                    x -= BLOCK_SIZE
                elif direction == Direction.DOWN:
                    y += BLOCK_SIZE
                elif direction == Direction.UP:
                    y -= BLOCK_SIZE

                new_pos = Point(x, y)
                if self._is_within_boundaries(new_pos) and new_pos not in self.snake[1:]:
                    queue.append(path + [(new_pos, direction)])

        return []

    def _is_within_boundaries(self, point):
        return 0 <= point.x < self.w and 0 <= point.y < self.h

    def _get_next_direction(self):
        root_state = GameState(self)
        root_node = Node(root_state)

        for _ in range(1000):
            node = root_node
            state = root_state

            # Selection
            while not node.is_fully_expanded() and node.children:
                node = node.select_child()
                state = state.next_state(node.state.direction)

            # Expansion
            if not node.is_fully_expanded() and not state.game.is_collision():
                node = node.expand()
                state = state.next_state(node.state.direction)

            # Simulation
            while not state.game.is_collision():
                random_direction = random.choice(list(Direction))
                state = state.next_state(random_direction)

            # Backpropagation
            score = state.game.score
            while node:
                node.backpropagate(score)
                node = node.parent

        best_child = max(root_node.children, key=lambda child: child.visits)
        return best_child.state.direction
