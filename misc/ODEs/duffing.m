       
[t,x] = ode45(@duffeqn, [0 200], [0.1; 0]);

plot(x(:,1),x(:,2))
title('Solution of Duffing Equation');

function dxdt = duffeqn(t,x)
a = 3;
b = 2;
c = 2;
d = 0.7;
w = 5;

dxdt = [x(2) ; c*cos(w*t) - a*x(1) - b*x(1)^3 - d*x(2) ];
end
