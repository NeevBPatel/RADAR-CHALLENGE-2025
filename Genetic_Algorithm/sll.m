function sll=sll(a,d,th0)
th0=th0*pi/180;
N=length(a);
af=zeros(1,360);
j=sqrt(-1);
for t=1:360
    s(t)=t;
    f(t)=t*pi/180;
    for n=1:N
        w=cos(f(t))-cos(th0);
        af(t)=af(t)+a(n)*exp(j*2*(n-1)*pi*d*w);
    end
    af(t)=abs(af(t));
end
p=findpeaks(af);
L=size(p);
for l=1:L(2)
    if(p(l)==max(af))
        p(l)=0;
    end
end
s0=max(p)
mini=max(af)
sll=20*log10(mini/s0)
af=af/mini;
af=20*log10(af);
figure(1);
plot(s,af);
title('Radiation Pattern ');
grid on;
axis([30,150,-30,0]);
xlabel('angle(theta)');
ylabel('af(dB)');

