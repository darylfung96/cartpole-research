import random

class ReplayMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = []

    def add_memory(self, current_state, reward, action, next_state, done):
        self.memory.append([current_state, reward, action, next_state, done])

        # we want to store less than the maximum size, if more than that we remove first one
        if len(self.memory) > self.max_size:
            self.memory.pop(0)

    def get_length(self):
        return len(self.memory)

    def get_batches(self, batch_size):
        assert batch_size <= len(self.memory)

        random.shuffle(self.memory)
        return self.memory[:batch_size]
