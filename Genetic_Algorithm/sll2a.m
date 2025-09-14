function [sll]=sll2a(A,th0,phi0,dx,dy);
[M N]=size(A);

th0 = (th0*pi)/180;
phi0 = (phi0*pi)/180;
phi=phi0;

lambda=1;
k=2*pi/lambda;
Bx=-k*dx*sin(th0)*cos(phi0);
By= -k*dy*sin(th0)*sin(phi0);
j = sqrt(-1);
AF = zeros(1,181);
s0=zeros(1,181);
for theta = -90:1:90
    s0(theta+91)=theta;
    % change degree to radian 
    deg2rad(theta+91) = (theta*pi)/180;
    
    for m=0:M-1
        for n=0:N-1
            AF(theta+91) = AF(theta+91) + A(m+1,n+1)*(exp(j*m*(k*dx*cos(phi)*sin(deg2rad(theta+91))+Bx)))*(exp(j*n*(k*dy*sin(phi)*sin(deg2rad(theta+91))+By))) ;
        end
    end
end
af = abs(AF);
p=findpeaks(af);
L=size(p);
for l=1:L(2)
    if(p(l)==max(af))
        p(l)=0;
    end
end
s=max(p);
mini=max(af);
sll=20*log10(mini/s)
af=af/mini;
af=20*log10(af);
figure(1);
plot(s0,af);
title('Radiation Pattern');
grid on;
axis([-90,90,-30,0]);
xlabel('angle(theta)');
ylabel('af(dB)');
