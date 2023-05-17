clear; clc;

if exist('OCTAVE_VERSION', 'builtin') ~= 0;
    saveOption = '-mat7-binary';
else
    saveOption = '-mat';
end
    

trainingFile = '1.5meters.offline.trace.txt';
testFile     = '1.5meters.online.trace.txt';
  
Macs = getMacs(trainingFile,testFile);
[database.trainingMacs,database.trainingLabels,database.trainingOthers] = generateMatrixes(trainingFile,Macs);
[database.testMacs,database.testLabels,database.testOthers] = generateMatrixes(testFile,Macs);

idx = 1;
for i = 2:size(database.trainingMacs,1)

  if    ((abs(database.trainingLabels(i,1)-database.trainingLabels(i-1,1))>0.001)||...
         (abs(database.trainingLabels(i,2)-database.trainingLabels(i-1,2))>0.001)||...
         (abs(database.trainingLabels(i,3)-database.trainingLabels(i-1,3))>0.001)||...
         (abs(database.trainingLabels(i,4)-database.trainingLabels(i-1,4))>2))
      idx = [idx, i];
  end
  i
end

idx2 = 1;
for i = 2:size(database.testMacs,1)
  if    ((abs(database.testLabels(i,1)-database.testLabels(i-1,1))>0.001)||...
         (abs(database.testLabels(i,2)-database.testLabels(i-1,2))>0.001)||...
         (abs(database.testLabels(i,3)-database.testLabels(i-1,3))>0.001)||...
         (abs(database.testLabels(i,4)-database.testLabels(i-1,4))>2))
    idx2 = [idx2, i];
  end
  i
end

database.trainingLabels(:,4) = 0;
database.trainingLabels(:,5) = 0;
database.testLabels(:,4) = 0;
database.testLabels(:,5) = 0;

database.trainingMacs   = database.trainingMacs(1:(idx(end)-1),:);
database.trainingLabels = database.trainingLabels(1:(idx(end)-1),:);
database.testMacs   = database.testMacs(1:(idx2(end)-1),:);
database.testLabels = database.testLabels(1:(idx2(end)-1),:);

idx = idx(1:(end-1));
idx2 = idx2(1:(end-1));

database0 = database;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


database.trainingLabels(:,4) = 0;
database.trainingLabels(:,5) = 0;
database.testLabels(:,4) = 0;
database.testLabels(:,5) = 0;

database.trainingMacs = database0.trainingMacs;
database.testMacs     = zeros(size(idx2,2)*10,size(database0.testMacs,2));
database.trainingLabels = database0.trainingLabels;
database.testLabels     = zeros(size(idx2,2)*10,5);

% if you want to create a different randompermutation, please remove the
% current file
if ~exist(['MANrandomperm.mat'])
  
  indexes_perm = zeros(size(idx2,2),109);
  
  for i = 1:size(idx2,2)
    indexes_perm(i,:)  = randperm(109);
  end
  
  save(['MANrandomperm.mat'],'indexes_perm',saveOption);
else
  load(['MANrandomperm.mat'],'indexes_perm');
end


for i = 1:size(idx2,2)
added = indexes_perm(i,1:10)

database.testMacs((i-1)*10+[1:10],:)   = database0.testMacs(idx2(i)+added,:);
database.testLabels((i-1)*10+[1:10],:) = database0.testLabels(idx2(i)+added,:);

end

save(['..' filesep() 'MAN1.mat'],'database',saveOption);


%%%%%%%%%%%%%

database.trainingMacs = zeros(size(idx,2)*10   ,size(database0.trainingMacs,2));
database.testMacs     = zeros(size(idx2,2)*10  ,size(database0.testMacs,2));
database.trainingLabels = zeros(size(idx,2)*10 ,5);
database.testLabels     = zeros(size(idx2,2)*10,5);

for i = 1:size(idx,2)
  for displacement = 0:9
    j = (i-1)*10+displacement+1;
    idx(i):1:(idx(i)+9)
    database.trainingMacs(j,:)   = averageMatrix(database0.trainingMacs(displacement+(idx(i):1:(idx(i)+9)),:));
    database.trainingLabels(j,:) = database0.trainingLabels(displacement+idx(i),:);
  end
end 

for i = 1:size(idx2,2)
  for displacement = 0:9
    j = (i-1)*10+displacement+1;
    database.testMacs(j,:)   = averageMatrix(database0.testMacs(displacement+(idx(i):1:(idx(i)+9)),:));
    database.testLabels(j,:) = database0.testLabels(displacement+idx(i),:);
  end
end 


save(['..' filesep() 'MAN2.mat'],'database',saveOption);


