       
[t,x] = ode45(@lorenzeqn, [0 100], [0.5; 0; 0]);

plot3(x(:,1),x(:,2),x(:,3));
title('Solution of Lorenz Equation');

function dxdt = lorenzeqn(t, x)
sigma = 6;
rho = 32;
beta = 2.5;

dxdt = [sigma*(x(2)-x(1)) ; x(1)*(rho-x(3)) - x(2) ; x(1)*x(2) - beta*x(3) ];
end
