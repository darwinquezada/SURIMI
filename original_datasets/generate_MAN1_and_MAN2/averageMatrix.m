function [output] = averageMatrix (matrix)

mask = matrix<100;
output = (sum(matrix.*mask)./(sum(mask)+(sum(mask)==0)))+100*(sum(mask)==0);

end