clc;
clear;
close all;

v = VideoWriter('cough', 'MPEG-4');
open(v);

r0 = [0; 0; 2];
T = 4.0;
dt = 1e-4;
v0 = 35;
vf = [0; 0; 0];
rho_i = 1000;
rho_a = 1.225;
Mtot = 0.00005;
Rm = 0.0001;
A = 0.9975;
Nc = [0; 1; 0];
Ac = [1; 0.5; 1];
muf = 1.8e-5;
g = [0; 0; -9.81];

mass = 0.0;
Ri = NaN(1,6100);
Mi = NaN(1,6100);
Pn = 0;
while mass < Mtot
    Pn = Pn + 1;
    e = -1 + 2*rand(1);
    Ri(Pn) = Rm * (1 + A * e);
    Mi(Pn) = rho_i*4/3*pi*Ri(Pn)^3;
    mass = mass + rho_i*4/3*pi*Ri(Pn)^3;
end

ni = zeros(3,Pn);
for i = 1:Pn
    etax = -1 + 2*rand(1);
    etay = rand(1);
    etaz = -1 + 2*rand(1);
    Ni = [Nc(1) + Ac(1) * etax; Nc(2) + Ac(2) * etay; Nc(3) + Ac(3) * etaz];
    ni(:,i) = Ni/vecnorm(Ni);
end

nTimeSteps = T/dt;
velParticle = v0 * ni;
posParticle = r0*ones(1,Pn);

% Init. 3D Fig.
fig1 = figure('pos', [0 200 1000 1000]);
h = gca;            
view(135, 30); 

axis equal;
grid on;

xlim([-1.5 1.5]);
ylim([0 2.5]);
zlim([0 3]);
xlabel('X[m]');
ylabel('Y[m]');
zlabel('Z[m]');
hold(gca, 'on');
count = 0;


Air = true(1,Pn);
Ai = pi*Ri.^2;  % Drag Reference Area
c = Ri(Air);
S = Ri(Air)./max(Ri) * 100;
s = S(:)';
fig_particles = scatter3(posParticle(1,:), posParticle(2,:), posParticle(3,:), s,c,'filled');
colormap

grid on;

for k = 1:nTimeSteps
    % DYNAMICS OF DROPLETS
%     posParticle(:,Air) = posParticle(:,Air) + dt*velParticle(:,Air);
    
    vdiff = vecnorm(vf-velParticle(:,Air),2,1);
    Re = (2*Ri(Air)*rho_a.*vdiff)./muf;
    Cd =    (24./Re).*(0 < Re & Re <= 1) + ...
            (24./(Re.^(0.646))).*(1 < Re & Re <= 400) + ...
            0.5.*(400 < Re & Re <= 3E5) + ...
            (0.000366*(Re.^(0.4275))).*(3E5 < Re & Re <= 2E6) + ...
            0.18.*(2E6 < Re);

    Fd = 0.5.*Cd.*rho_a.*Ai(Air).*vdiff.*(vf-velParticle(:,Air));
    Fg = Mi(Air).*g;

    Ftot = Fd + Fg;

    posParticle(:,Air) = posParticle(:,Air) + velParticle(:,Air)*dt; % Use Forward Euler to update position
    velParticle(:,Air) = velParticle(:,Air) + dt*(Ftot./Mi(Air)); % Use Forward Euler to update velocity
    below = find(posParticle(3,:) < 0); % Check if droplets have hit the print bed
    posParticle(3,below) = 0; % Reset droplets height to print bed
    Air(below) = false; % Update flag array to indicate droplets on bed
    
    if mod(k,100) == 0 % update movie frame
        count = count + 1;
        view(135+count/3, 30);
        figure(1)
        set(fig_particles, ...
        'xData', posParticle(1,:), ...
        'yData', posParticle(2,:), ...
        'zData', posParticle(3,:));
        frame = getframe(gcf);
        writeVideo(v,frame);
    end
end

close(v);