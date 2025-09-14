clc;
clear all;
N=8; % bits in a gene
M=N; 
last=100; 
M2=M/2;
Gene=round(rand(M,N)*127);
iteration = zeros(1,last);
cost_val = zeros(1,last);
for ib=1:last
    %Gene(:,8)=0;
    iteration(ib) = ib;
    for el=1:1:N
    cost(el)=sll(Gene(el,:),0.5,90);
    cost_val(ib) = max(cost);
    end
    [cost,ind]=sort(cost);
    Gene=Gene(ind(M2+1:N),:);
    cross=ceil((N-1)*rand(M2,1));
    for ic=1:2:M2
        Gene(M2+ic,1:cross)=Gene(ic,1:cross);
        Gene(M2+ic,cross+1:N)=Gene(ic+1,cross+1:N);
        Gene(M2+ic+1,1:cross)=Gene(ic+1,1:cross);
        Gene(M2+ic+1,cross+1:N)= Gene(ic,cross+1:N);
    end
    ix=ceil(M*rand)
    iy=ceil(N*rand)
    Gene(ix,iy)=127-Gene(ix,iy)
end