function SmoothVoltageData(varargin);
%
% SmoothVoltageData(subsample,Np,sigO,R_input,T)
%
% Model-based smoothing of noisy voltage measurements. This code is released in
% conjunction with the following publication 
%
%   Huys QJ and Paninski L (2007): Model-based filtering of, and parameter
%   estimation from, noisy biophysical observations. 
%   (available at http://www.gatsby.ucl.ac.uk/~qhuys/pub.html)
%
% and can be downloaded from 
%
%   http://www.gatsby.ucl.ac.uk/~qhuys/code.html
%
% This function is the main function and contains, at the bottom, the two other
% functions needed. It sets up the cell and runs a particle smoother on data
% corrupted by Gaussian noise of variance sigO. 
%
% It takes as optional arguments [default values in brackets]
% 
%  subsample [7]     temporal subsampleampling (1 = no subsampleampling, subsample>1: 
%                    only see every subsample sample) 
%  Np        [30]    number of particles 
%  sigO      [30]    observation noise variance
%  R_input   [1]     input resistance to injected current
%  T         [1000]  length of the simulated trace
% 
% and produces figure 2 of the above mentioned paper. If no arguments are
% passed, it uses the same parameters as in the paper figure. 
% 
% It calls HH.m, which contains the cellular dynamics both to generate the data
% and to then run the particle filter (ie here knowledge fo the true dynamics
% are assumed known). When plotting, it also calls a custom plotting command
% MYEB2.m, which plots data as a mean plus a shaded region of 1 standard
% deviation width. 
%
% Copyright Quentin Huys 2007 
% Center for Theoretical Neuroscience, Columbia University 
% user = qhuys
% domain = gatsby
% Email: user@domain.ucl.ac.uk (replace user and domain)
% Web: http://www.gatsby.ucl.ac.uk/~qhuys

%----------------------------------------------------------------------------
%     PARAMETERS ASSIGNMENT
% 
%     These are the main parameters that can be set when calling the function.
%     Nothing else should be changed, unless you know what you're doing. 
%
%     Default settings: 
%
% 	      subsample=7;       % subsampleampling
% 	      Np=30;             % number of particles
% 	      sigO=30;           % voltage measurement noise
% 	      R_input=1;         % input resistance
% 	      T=1000;            % length of recording
%
%----------------------------------------------------------------------------
if ~exist('varargin'); nargin=0;
else  nargin = length(varargin);
end
parname = {'subsample','Np','sigO','R_input','T'};
for k=1:nargin
	eval([parname{k} ' = ' num2str(varargin{k}) ';']);
end
pardefaults = [7;30;30;1;1000];
for k=nargin+1:5;
	eval([parname{k} ' = ' num2str(pardefaults(k)) ';']);
end

%----------------------------------------------------------------------------
%		Setup cell 
%		These parameters should not be changed
%----------------------------------------------------------------------------
nc=1;				      % number of compartments
Nc=4*nc;			      % number of parameters
dt=.02;              % integration time step
Ivar = 80;           % size of square current pulses injected
E=[50 -80 -55];      % reversal potentials of channels
sigV=10*sqrt(dt);     % evolution voltage noise
res_th=Np/2;         % resampling threshold

% setting up some things we'll need later
tv=(1:T)*dt;         % vector with time points
subsamplet=(mod(1:T,subsample)==0);
ind1=(1:nc);
ind2=(1:nc)+nc;
ind3=(1:nc)+2*nc;
ind4=(1:nc)+3*nc;

g0=[120 20 3];
R0=zeros(nc);
R0(1,1)=R_input;

% generate input current
tm=cos((1:T)/T*10*pi+pi);
I(1,:)=Ivar*(((tm>0)));I(2:nc,:)=0;

% prepare for data simulation -- assign all cellular parameters to one structure
g=g0;R=R0;par0=[g0,R0];
hhs.zv=zeros(Nc,1);
hhs.nc=nc;
hhs.Nc=Nc;
hhs.dt=dt;
hhs.E=E;
hhs.g=g0;
hhs.R=R0;
hhs.oo=zeros(nc,6);

%----------------------------------------------------------------------------
%		GET DATA 
%----------------------------------------------------------------------------
y=zeros(nc,T);
v=zeros(Nc,T);
% initial conditions
v(ind1,1)=-65; v(ind2,1)=.30; v(ind3,1)=.08; v(ind4,1)=.65;

% generate data 
for t=2:T; 
	v(:,t) = v(:,t-1) + dt*hh(v(:,t-1),I(:,t-1),hhs) + sigV*[randn(nc,1);zeros(Nc-nc,1)];
end

% add measurement noise
y = v(1:nc,:) + sigO*randn(nc,T);

%----------------------------------------------------------------------------
%		E step -- particle filter / smoother 
%----------------------------------------------------------------------------
fprintf('......... Running particle filter \n');

xp=zeros(Nc,Np,T);
xp(ind1,:,1)=-65;
xp(ind2,:,1)=.30;
xp(ind3,:,1)=.08;
xp(ind4,:,1)=.65;
xmf=zeros(Nc,T);
xmf(:,1)=xp(:,1,1);
xxf=zeros(Nc,T);
xpp=zeros(Nc,Np);
wfs=ones(Np,T)/Np;
rsit=0;

zv=zeros(Nc-nc,1);
zzv=zeros(Nc,1);
znp=zeros(Np,1);
flatw=1/Np*ones(1,Np);

rv=randn(Nc,T,Np);	% random numbers to evolve particles
rs=rand(Np,T)/Np + repmat([0:Np-1]'/Np,1,T);% random numbers for stratified resampling 

%..............particle filter 
for t=2:T
	for np=1:Np											% forward step
		xp(:,np,t) = xp(:,np,t-1) + dt*hh(xp(:,np,t-1),I(:,t-1),hhs) + sigV*[rv(nc,t,np);zv];
		i=find(xp(2:end,np,t)<0);xp(i+1,np,t)=0;
		i=find(xp(2:end,np,t)>1);xp(i+1,np,t)=1;
	end

	%.........weights 
	if mod(t,subsample)~=0	% don't update weights if no obs
		wfs(:,t)=wfs(:,t-1);
	else 
		for np=1:Np
			dist(np)=sum(-1/2*(y(:,t)-xp(ind1,np,t)).^2./sigO^2);
		end
		edist=exp(dist);
		if sum(edist)==0; % if get rounding errors, only keep closest particle
			wchosen=find(dist==max(dist));
			wfs(:,t) = znp;
			wfs(wchosen,t) = 1;
		else 
			wfs(:,t) = wfs(:,t-1).*edist'; % p(x_t|x_{t-1},uB) 
			swfs=sum(wfs(:,t));
			if swfs>0; wfs(:,t) = wfs(:,t)/sum(wfs(:,t));	
			else 
				wchosen=find(dist==max(dist));
				wfs(:,t) = znp;
				wfs(wchosen,t) = 1;
			end
		end
		% ..................resampling......................................
		if 1/(wfs(:,t)'*wfs(:,t))<res_th;
			r=rs(:,t);% stratified resampling 
			[foo,ind] = histc(r,[0 cumsum(wfs(:,t)')]);
			xp(:,:,t) = xp(:,(ind),t);				% copy particles		
			wfs(:,t) = flatw;			% reset weights
			rsit=rsit+1;
		end
		if ~mod(t,100);fprintf('............ fwd particles t = %i, resampled %i times \r',t,rsit);end
		xpw=xp(:,:,t).*repmat(wfs(:,t)',[Nc 1]);
		xmf(:,t) = sum(xpw,2);
		xxf(:,t) = sum(xp(:,:,t).^2.*repmat(wfs(:,t)',[Nc 1]),2)-xmf(:,t).^2;
	end
end

%.........particle smoother 
fprintf('\n............ particle smoother and sufficient stats')
xx=zeros(Nc);
PxP=zeros(Np);
wij = repmat(wfs(:,T),[1 Np])/Np;
M=zeros(Nc);
f1=zeros(Nc,1);
f2=zeros(Nc,1);
sxpz=zeros(Np,Np);

for t=T:-1:2
	% backwards weights 
	for np=1:Np											% forward step
		xpp(:,np,t) = xp(:,np,t-1) + dt*hh(xp(:,np,t-1),I(:,t-1),hhs);
		i=find(xpp(2:end,np,t)<0);xpp(i+1,np,t)=0;
		i=find(xpp(2:end,np,t)>1);xpp(i+1,np,t)=1;
	end
	% P(x^i_t|x^j_t-1,u,B)
	sxp=sxpz;
	%for k=1:Nc
	for k=1:nc 	% watch out: this means we're neglecting the transition probs of the gates! 
		         % in practice this doesn't have much effet
		sxp=sxp + (repmat(xp(k,:,t)',1,Np) - repmat(xpp(k,:,t),Np,1)).^2/sigV^2;
	end
	PxP = exp(-1/2*sxp);
	tmp=sq(PxP).*... 									% P(x_t|x_t-1)
		repmat(wfs(:,t-1)',[Np 1]);				% P(x_t-1|y^t-1)
	wij=tmp./repmat(sum(tmp,2),[1 Np]).*...	% P(x_t,x_t-1|y^t-1) / P(x_t|y^t-1)
		repmat(sum(wij,1)',[1 Np]);				% P(x_t|y^T)

	wbs = sum(wij,2);
	wps = sum(wij,1);

	% expected sufficient statistics
	xpw=xp(:,:,t).*repmat(wbs',[Nc 1]);
	xm(:,t) = sum(xpw,2);
	xv(:,t) = sum(xp(:,:,t).^2.*repmat(wbs',[Nc 1]),2)-xm(:,t).^2;

	if ~mod(t,100);fprintf('............ bwd particles t = %i\r',t);end

end

%----------------------------------------------------------------------------
%      Plot figure 2 from the paper 
%----------------------------------------------------------------------------
clf;k=1;
subplot(5,1,k);k=k+1;
	ind=subsample:subsample:T;
	plot(tv(ind),y(1,ind),'o','color',.3*ones(1,3));
	axis tight
	set(gca,'xticklabel',[],'fontsize',14)
	ylabel({'Voltage','[mV]'});
subplot(5,1,k);k=k+1;
	ind=2:T;
	plot(tv(ind),sq(xp(1,:,ind))','color',.6*ones(1,3));
	axis tight
	set(gca,'xticklabel',[],'fontsize',14)
	ylabel({'Voltage','[mV]'});
subplot(5,1,k);k=k+1;
	plot(tv(ind),1./sum(wfs(:,ind).^2),'k')
	axis tight
	set(gca,'xticklabel',[],'fontsize',14)
	ylabel('N_{eff}');
subplot(5,1,k);k=k+1;
	myeb2(tv(ind),xm(1,ind),sqrt(xv(1,ind)));
	hold on
	plot(tv(ind),v(1,ind),'k--','linewidth',2);
	hold off
	axis tight
	set(gca,'xticklabel',[],'fontsize',14)
	ylabel({'Voltage','[mV]'});
subplot(5,1,k);k=k+1;
	myeb2(tv(ind),xm(2:end,ind)',sqrt(xv(2:end,ind))');
	hold on
	plot(tv(ind),v(2:end,ind),'k--','linewidth',2)
	hold off
	axis tight
	box on
	set(gca,'fontsize',14)
	ylabel('Open prob');
	xlabel('Time [ms]');


function [dV] = hh(V,I,h);
% 
% Simulates a cell with typical Hodgkin-Huxley channels. Uses Euler integration, so we need the
% timestep to be small to prevent instabilities. 
%

dV=h.zv;

ind1=(1:h.nc);
ind2=(1:h.nc)+h.nc;
ind3=(1:h.nc)+2*h.nc;
ind4=(1:h.nc)+3*h.nc;

dV(ind1) = h.g(:,1).*V(ind2).^3.*V(ind3).*	(h.E(1)-V(ind1)) ...		% HH channels
			+ h.g(:,2).*V(ind4).^4.*				(h.E(2)-V(ind1)) ...		
			+ h.g(:,3).*								(h.E(3)-V(ind1)) ...		% leak channel
			+ h.R*I;

%.......................... kinetics ...........................................

h.oo(:,1)	=0.1*(V(ind1)+35)./(1-exp(-(V(ind1)+35)/10));% a_m % HH gates -- modified
h.oo(:,2)	=0.07*exp(-(V(ind1)+50)/20);			% a_h
h.oo(:,3)	=0.01*(V(ind1)+55)./(1-exp(-(V(ind1)+55)/10));     	% a_n
h.oo(:,4)	=4*exp(-(V(ind1)+65)/18);				% b_m
h.oo(:,5)	=1./(exp(-(V(ind1)+35)/10)+1);			% b_h
h.oo(:,6)	=0.125*exp(-(V(ind1)+65)/80);			% b_n


%.......................... gate diffeq's.......................................

dV(ind2)  = h.oo(:,1).*(1-V(ind2)) - h.oo(:,4).*V(ind2);
dV(ind3)  = h.oo(:,2).*(1-V(ind3)) - h.oo(:,5).*V(ind3);
dV(ind4)  = h.oo(:,3).*(1-V(ind4)) - h.oo(:,6).*V(ind4);



function h=myeb2(X,Y,varargin);
%
% myeb2(X,Y,varargin);
%
% This function makes nice coloured, shaded error bars. Exactly what
% it does depends on Y, and on whether you give it one or two inputs. 
%
% If you only pass it Y, and no other arguments, it assuemd you're
% giving it raw data. 
%
%		myeb2(X,Raw_Data)
%
% 	.) if Y is 2D array, it will then plot mean(Y) with errorbars given
% 	by std(Y). In this case there is only one mean vector with its
% 	errorbars. 
% 
%	.) if Y is 3D array, it will plot size(Y,3) lines with the
%	associated errorbars. Line k will be mean(Y(:,:,k)) with errorbars
%	given by std(Y(:,:,k))
%
% If you pass it 2 arguments, each has to be at most 2D. 
%
%		myeb(X,mu,std)
%
% 	.) if mu and std are 1D, it just plots one line given by mu with a
% 	shaded region given by std. 
%
%	.) if mu and std are 2D, it will plot size(Y,2) lines in the
%	standard sequence of colours; each line mu(:,k) will have a shaded
%	region in the same colour, but less saturated given by std(:,k)
%
%
% Quentin Huys
% August 14th, 2007
% Center for Theoretical Neuroscience, Columbia University
% Email: qhuys [at] n e u r o theory [dot] columbia.edu
% (just get rid of the spaces, replace [at] with @ and [dot] with .)



col=[0 0 1; 0 .5 0; 1 0 0; 0 1 1; 1 0 1; 1 .5 0; 1 .5 1];
ccol=col+.8; ccol(ccol>1)=1;

if length(size(Y))==3 & size(Y,2)~=length(X); 
	error('X and size(Y,2) must match');
end

holdstatus=get(gca,'NextPlot');

if length(varargin)==0;

	if length(size(Y))==2 
		m=mean(Y);
		s=std(Y);
		ind1=1:length(m);
		ind2=ind1(end:-1:1);
		th=fill([X(ind1) X(ind2)],[m-s m(ind2)+s(ind2)],.6*ones(1,3));
		set(th,'edgecolor',.6*ones(1,3));
		hold on; 
		h=plot(X(ind1),m,'linewidth',2);
		set(gca,'NextPlot',holdstatus);
	elseif length(size(Y))>2 
		if strcmpi(holdstatus,'replace');cla;end
		hold on; 
		ind1=1:size(Y,2);
		ind2=ind1(end:-1:1);
		if size(Y,3)>8; col=jet(size(Y,3));ccol=col+.8; ccol(ccol>1)=1;end
		for k=1:size(Y,3)
			m=mean(Y(:,:,k));
			s=std(Y(:,:,k));
			th=fill([X(ind1) X(ind2)],[m-s m(ind2)+s(ind2)],ccol(k,:));
			set(th,'edgecolor',ccol(k,:))
		end
		for k=1:size(Y,3)
			m=mean(Y(:,:,k));
			s=std(Y(:,:,k));
			h(k)=plot(X(ind1),m,'linewidth',2,'color',col(k,:));
		end
		set(gca,'NextPlot',holdstatus);
	end

elseif length(varargin)==1;

	m=Y;
	s=varargin{1};
	if length(size(Y))>2; error;
	elseif min(size(Y))==1;
		if size(m,1)>1; m=m';s=s';end
		ind1=1:length(m);
		ind2=ind1(end:-1:1);
		th=fill([X(ind1) X(ind2)],[m-s m(ind2)+s(ind2)],.6*ones(1,3));
		hold on; 
		set(th,'edgecolor',.6*ones(1,3));
		plot(X(ind1),m,'linewidth',2);
		set(gca,'NextPlot',holdstatus);
	else 
		ind1=(1:size(Y,1));
		ind2=ind1(end:-1:1);
		if strcmpi(holdstatus,'replace');cla;end
		hold on; 
		if size(Y,2)>8; col=jet(size(Y,2));ccol=col+.8; ccol(ccol>1)=1;end
		for k=1:size(Y,2)
			mm=m(:,k)';
			ss=s(:,k)';
			th=fill([X(ind1) X(ind2)],[mm-ss mm(ind2)+ss(ind2)],ccol(k,:));
			set(th,'edgecolor',ccol(k,:))
		end
		for k=1:size(Y,2);
			mm=m(:,k)';
			ss=s(:,k)';
			h(k)=plot(X(ind1),mm,'linewidth',2,'color',col(k,:));
		end
		set(gca,'NextPlot',holdstatus);
	end
end


function y=sq(x)
y=squeeze(x);


