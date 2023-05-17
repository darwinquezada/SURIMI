function [fp,labels,others] = generateMatrixes (filename, macs)

fid          = fopen (filename);
tmp          = textscan(fid,'%s','delimiter','\n');
num_of_lines =  size(tmp{1},1);
fclose(fid);

fid          = fopen (filename);
contents     = {};
num_of_lines

j = 0;
k = 0;

fp     = 100 + zeros(num_of_lines,numfields(macs));
labels = zeros(num_of_lines,4);
others = zeros(num_of_lines,1);

for i=1:num_of_lines
  
  fpline =  fgets (fid);
  if size(fpline,2)>0 && (fpline(1)~='#')
    
    j = j+1;
    
    contentsRaw{j} = fpline;    
    
    l = 0;
    fpelements = strsplit(fpline,';');
    for idx = 1:size(fpelements,2)
      fpelement = fpelements{idx};
      if strfind(fpelement,'t=')
        others(j,1) = str2num(fpelement(3:end));  
      elseif strfind(fpelement,'pos=')
        pos = strsplit(fpelement(5:end),','); 
        labels(j,1) = str2num(pos{1}); 
        labels(j,2) = str2num(pos{2}); 
        labels(j,3) = str2num(pos{3});         
      elseif strfind(fpelement,'degree=')
        labels(j,4) = str2num(fpelement(8:end));            
      elseif strfind(fpelement,'id=')
        %others(j,2) = str2num(fpelement(4:end));        
      else
 
        rssVec =  strsplit(fpelement(19:end),',');
        fp(j,macs.(['MAC' strrep(fpelement(1:17),':','')])) = str2num(rssVec{1});

        
      end
    end
    
  end
  if ~mod(i,100)
    i
  end
end
j
k
fclose(fid);
end
