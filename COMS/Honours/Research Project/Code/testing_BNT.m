%Creates the Cloudy-Sprinkler-Rain-Wet model

N = 4;  %Number of nodes
dag = zeros(N,N);   %The graph, encodes the connections
C = 1; S = 2; R = 3; W = 4; %Each node
dag(C,[R S]) = 1;
dag(R,W) = 1;
dag(S,W) = 1;

false = 1; true = 2;
node_sizes = 2*ones(1,N); %How many states for each variable

%Create the Bayesian Network
bnet = mk_bnet(dag, node_sizes, 'names', {'Cloudy','Sprinkler','Rain','Wet Grass'}, 'discrete', 1:N);
names = bnet.names;

%Create CPDs
bnet.CPD{C} = tabular_CPD(bnet, C, [0.5 0.5]);
bnet.CPD{R} = tabular_CPD(bnet, R, [0.8 0.2 0.2 0.8]);
bnet.CPD{S} = tabular_CPD(bnet, S, [0.5 0.9 0.5 0.1]);
bnet.CPD{W} = tabular_CPD(bnet, W, [1 0.1 0.1 0.01 0 0.9 0.9 0.99]);

%Junction tree engine
engine = jtree_inf_engine(bnet);

%What we know
evidence = cell(1,N);
evidence{W} = true;

%Enter evidence to engine
[engine, loglik] = enter_evidence(engine, evidence);

%Calculate P(S=true|W=true)
marg = marginal_nodes(engine, S);
p = marg.T(true)
