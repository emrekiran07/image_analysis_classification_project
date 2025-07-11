function H=hu_moments(I)
%function H=hu_moments(I)
%
% Compute eight moment invariants, according to Hu, and Flusser and Suk
% See: https://en.wikipedia.org/wiki/Image_moment#Rotation_invariants
%
% Are invariant to rotation and scaling
% H(7) and H(8) change sign on flips (parity)
%
%Author: Joakim Lindblad

% Copyright (c) 2020, Joakim Lindblad


[i,j,v]=find(I);
if isrow(i)
	i=i(:);j=j(:);v=v(:); 
end

%Zero moment
mu00=sum(v); %M_00=mu_00
ctr=sum([i,j].*v,1)./mu00;

%Centralize
i=i-ctr(1);
j=j-ctr(2);

%Scale invariant central moments of order {2,3}
eta=nan(4,4);
for p=0:3
	for q=max(0,2-p):3-p %Only compute when needed
		eta(p+1,q+1)=sum(i.^p .* j.^q .* v) / mu00.^(1+(p+q)/2);
	end
end

%Moment invariants
H(1)=eta(3,1)+eta(1,3);
H(2)=(eta(3,1)-eta(1,3))^2 + 4*eta(2,2)^2;
H(3)=(eta(4,1)-3*eta(2,3))^2 + (3*eta(3,2)-eta(1,4))^2;
H(4)=(eta(4,1)+eta(2,3))^2 + (eta(3,2)+eta(1,4))^2;
H(5)=(eta(4,1)-3*eta(2,3))* (eta(4,1)+eta(2,3))*( (eta(4,1)+eta(2,3))^2 - 3*(eta(3,2)+eta(1,4))^2 ) + ...
    (3*eta(3,2)-eta(1,4)) * (eta(3,2)+eta(1,4))*( 3*(eta(4,1)+eta(2,3))^2 - (eta(3,2)+eta(1,4))^2 );
H(6)=(eta(3,1)-eta(1,3))*( (eta(4,1)+eta(2,3))^2 - (eta(3,2)+eta(1,4))^2 ) + 4*eta(2,2)*(eta(4,1)+eta(2,3))*(eta(3,2)+eta(1,4));
H(7)=(3*eta(3,2)-eta(1,4))* (eta(4,1)+eta(2,3))*( (eta(4,1)+eta(2,3))^2 - 3*(eta(3,2)+eta(1,4))^2 ) - ...
     (eta(4,1)-3*eta(2,3))* (eta(3,2)+eta(1,4))*( 3*(eta(4,1)+eta(2,3))^2 - (eta(3,2)+eta(1,4))^2 );
H(8)=eta(2,2)*( (eta(4,1)+eta(2,3))^2 - (eta(3,2)+eta(1,4))^2 ) - (eta(3,1)-eta(1,3))*(eta(4,1)+eta(2,3))*(eta(3,2)+eta(1,4));
