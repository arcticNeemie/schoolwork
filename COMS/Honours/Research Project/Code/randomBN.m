%
%Create a BN with random CPDs, then sample from it and use it to learn a
%new one
%

N = 6; %Number of Nodes
dag = zeros(N,N); %A graph for encoding connections
A = 1; B = 2; C = 3; D = 4; E = 5; F = 6; %For convenience, we can label nodes with letters
%Connections
dag(A,[B C]) = 1;
dag(B,[D E]) = 1;
dag(C,F) = 1;
dag(E,F) = 1;

false = 1; true = 2; %For convenience, can refer to 1 and 2 as true or false
node_sizes = 2*ones(1,N); %Set number of possibilities per node

%Create the BN
bnet = mk_bnet(dag, node_sizes, 'names', {'A','B','C','D','E','F'}, 'discrete', 1:N);
names = bnet.names;

%Create CPDs
seed = 12345;
rand('state',seed); %For repeatable results
%randn('state',seed);

bnet.CPD{A} = tabular_CPD(bnet, A);
bnet.CPD{B} = tabular_CPD(bnet, B);
bnet.CPD{C} = tabular_CPD(bnet, C);
bnet.CPD{D} = tabular_CPD(bnet, D);
bnet.CPD{E} = tabular_CPD(bnet, E);
bnet.CPD{F} = tabular_CPD(bnet, F);

%Visualize
%[~,~,h] = draw_graph(bnet.dag);

%Junction tree engine
engine = jtree_inf_engine(bnet);

%What we know
evidence = cell(1,N);
evidence{A} = true;

%Enter evidence to engine
[engine, loglik] = enter_evidence(engine, evidence);

%Calculate P(B=true|A=true)
marg = marginal_nodes(engine, B);
p = marg.T(true);

%Generate Data
nsamples = 20000;
samples = cell(N, nsamples);
for i=1:nsamples
  samples(:,i) = sample_bnet(bnet);
end

%Learn parameters using MLE
bnet2 = learn_params(bnet, samples);

%Matlab hack needed to view learned parameters
CPT2 = cell(1,N);
CPT = cell(1,N);
for i=1:N
  s=struct(bnet2.CPD{i});  % violate object privacy
  t=struct(bnet.CPD{i});
  CPT2{i}=s.CPT;
  CPT{i}=t.CPT;
end

dispcpt(CPT2{F})
disp("-")
dispcpt(CPT{F})

