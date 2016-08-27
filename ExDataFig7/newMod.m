function [ArMod SkelMod]=newMod(A,S,M,P)
%newMod(A,S,M) takes input values of adjacency matrix=A, skeleton list=S for
%said matrix and the modularity_dir(A) =M
%then orders the rows and columns by modularity group
n=length(S);
ngroups=max(M);
noutof=[];
for i=(1:ngroups)
    noo=length(find(M~=i));
    while (isempty(find(noutof==noo))==0)
        noo=noo+1;
    end
    noutof=[noutof;noo];
end
OtherGrps=[];

for i=(1:n)
    OtherGrps=[OtherGrps; noutof(M(i))];
end

FunctVals=NaN(n,9);
for i=1:n
    if ismember(S(i),P(:,1))
        FunctVals(i,1:9)=P(P(:,1)==S(i),2:10);
    end
end
    
SkelModa=[S,M,reshape(1:n,n,1) OtherGrps FunctVals];
% ReOrderedSkel=sortrows(SkelModa,4);
ReOrderedSkel=sortrows(SkelModa,[4 -6]);
ArMod=[];
tempAr=[];
for i=(1:n)
    index=ReOrderedSkel(i,3);
    tempAr=[tempAr,A(:,index)];
end
for i=(1:n)
    index=ReOrderedSkel(i,3);
    ArMod=[ArMod;tempAr(index,:)];
end
SkelMod=ReOrderedSkel;
end