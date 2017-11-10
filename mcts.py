import numpy


class Node:
    def __init__(self, state, action=0, priors=[], parent=None):
        print("init", state, action, priors)
        self.state = state
        self.action = action
        self.children = [None,]*len(priors)
        self.parent = parent
        self.priors = priors
        self.value = 0
        self.visits = 0

    def add_child(self, state, action, ):
        self.children.append(Node(state))

    def update(self, value):

        self.value = (self.value*self.visits + value) / (self.visits+1)
        self.visits += 1

    def val(self, action):
        if self.children[action] == None:
            return self.priors[action]
        else:
            print("c",self.children[action].value,self.children[action].visits)
            return self.children[action].value + self.priors[action]/(self.children[action].visits +1)

    def best_child(self):
        best_val = self.val(0)
        best_action = 0
        for action in range(1,len(self.priors)):
            if self.val(action) > best_val:
                best_action = action
                best_val = self.val(action)
        print("best",action, "state",self.state, self.priors)
        return best_action

    def pi_est(self):
        pi_est = [ self.children[action].visits if self.children[action]
                                                   is not None else 0 for action in range(len(self.priors)) ]


        #for action in range(1,len(self.priors)):
        print(pi_est)
        return pi_est

    def __del__(self):
        for c in self.children:
            del c


class add_div_problem:
    def __init__(self):
        self.state = 3.14

    def update_state(self, state):
        self.state = state

    def step(self, action, old_state):
        print("mstate",self.state, "nstate",old_state)
        assert self.state == old_state

        print("step", self.state, action)
        rew = -(self.state-5.0)**2
        if action<0.1:
            new_state = self.state-0.75
        else:
            new_state = self.state*1.25

        print("step", self.state, new_state, action)
        self.state = new_state
        return new_state, rew


search_iters = 10
search_depth = 10


def eval(state):
    rew = -(state-5.0)**2
    priors = [state>5.0, state<5.0]
    priors = [p*0.5+0.5 for p in priors]
    return priors, rew


def select(model,tree):
    print("Select")
    n = tree
    for depth in range(0,search_depth):
        action = n.best_child()
        print("best a",action, "d", depth)

        if n.children[action] is None:
            expand(model, n,action)

        n = n.children[action]
    return n


def expand(model, node,action):
    print("Expand")
    model.update_state(node.state)
    (new_state, rew) = model.step(action,node.state)

    priors, value = eval(node.state)

    node.children[action] = Node(state=new_state, action=action, priors=priors, parent=node)



def backup(node):
    n = node
    p, v  = eval(node.state)

    while n.parent is not None:
        print(n.state)
        n.update(v)
        n = n.parent


def follow_tree(tree, temperture):
    n = tree

    pi_est = n.pi_est()
    if temperture < 0.1:
        action = numpy.argmax(pi_est)
    else:
        raise NotImplementedError

    return n.children[action]


def print_best_path(tree):
    n = tree
    while n is not None:

        pi_est = n.pi_est()

        action = numpy.argmax(pi_est)

        print("state", n.state, "v", n.visits, n.pi_est(), action)

        n = n.children[action]


def training_iteration():
    state0 = 3.14
    model = add_div_problem()
    priors, value = eval(state0)

    tree = Node(state0, priors=priors)
    #expand(model, tree,0)
    for i in range(search_iters):
        model.update_state(tree.state)

        n = select(model,tree)
        backup(n)

    print_best_path(tree)
    tree = follow_tree(tree,0)


def main():
    training_iteration()




if __name__ == "__main__":
    main()

