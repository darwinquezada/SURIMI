function [Macs] = getMacs (training, test)

% fid          = fopen (training);
% num_of_lines1= fskipl(fid, Inf);
% fclose(fid);
% fid          = fopen (test);
% num_of_lines2= fskipl(fid, Inf);
% fclose(fid);

fid          = fopen (training);
tmp          = textscan(fid,'%s','delimiter','\n');
num_of_lines1=  size(tmp{1},1);
fclose(fid);
fid          = fopen (test);
tmp          = textscan(fid,'%s','delimiter','\n');
num_of_lines2=  size(tmp{1},1);
fclose(fid);

fid          = fopen (training);
Macs    = {};

counter = 0;
for i=1:num_of_lines1

  fpline =  fgets (fid);
  if size(fpline,2)>0 && (fpline(1)~='#')
    
    fpelements = strsplit(fpline,';');
    for idx = 1:size(fpelements,2)
      fpelement = fpelements{idx};
      if strfind(fpelement,'t=')
      elseif strfind(fpelement,'pos=')              
      elseif strfind(fpelement,'degree=')
      elseif strfind(fpelement,'id=')
      else
      
        if ~isfield(Macs,['MAC' (strrep(fpelement(1:17),':',''))])
        counter = counter+1;
        Macs.(['MAC' (strrep(fpelement(1:17),':',''))]) = counter;
        end
        
      end
    end
   end
    i
end
fclose(fid);

fid          = fopen (test);
for i=1:num_of_lines2

  fpline =  fgets (fid);
  if size(fpline,2)>0 && (fpline(1)~='#')
    
    fpelements = strsplit(fpline,';');
    for idx = 1:size(fpelements,2)
      fpelement = fpelements{idx};
      if strfind(fpelement,'t=')
      elseif strfind(fpelement,'pos=')              
      elseif strfind(fpelement,'degree=')
      elseif strfind(fpelement,'id=')
      else
        
        if ~isfield(Macs,['MAC' (strrep(fpelement(1:17),':',''))])
        counter = counter+1;
        Macs.(['MAC' (strrep(fpelement(1:17),':',''))]) = counter;
        end
        
      end
    end
   end
    
end
fclose(fid);

end
