classdef BayesFilter < handle
    properties
        x_ll % right limit
        x_rl % right limit
        x0
        sigma_z0
        
        seed = 100
        spaces = 1000
        
    end
    
    methods
        function obj = BayesFilter(x_ll,x_rl, x0, sigma_z0)
            obj.x_ll = x_ll;
            obj.x_rl = x_rl;
            obj.x0 = x0;
            obj.sigma_z0 = sigma_z0;
        end
        
        function setRandom(obj)
            rng(obj.seed)
        end
        
        function setSeed(obj, seed)
           obj.seed = seed;
        end
        
        function setSpaces(obj, spaces)
           obj.spaces = spaces;
        end
        
        function runSimulation(obj)
            
            obj.setRandom();
            
            dx = (obj.x_rl - obj.x_ll)/obj.spaces;
            x = obj.x_ll:dx:obj.x_rl;

            z = obj.x0 + obj.sigma_z0 * randn;

            p_priori = 1/(obj.x_rl - obj.x_ll) * ones(size(x));
            p_posteriori = BayesFilter.normGaussian(z(end), obj.sigma_z0, x);
            
            p_z = sum(p_posteriori .* p_priori * dx);
            p_bayes = p_posteriori .* p_priori/p_z;
            
            E_bayes  = [];
            sigma_bayes = [];
            k = 1;
            
            %stop_sigma = 0.05 * obj.sigma_z0;
            stop_sigma = 0.30 * obj.sigma_z0;
            
            E_priori = sum(x .* p_priori * dx);
            max_priori = max(p_priori);
            max_posteriori = 1/sqrt(2 * pi * obj.sigma_z0^2);
            E_bayes = [E_bayes, sum(x .* p_bayes *dx)];
            var_bayes = sum((x-E_bayes(end)).^2 .* p_bayes * dx);
            sigma_bayes = [sigma_bayes, sqrt(var_bayes)];
            max_bayes = max(p_bayes);
            
            while sigma_bayes > stop_sigma
                k = k + 1;
                p_priori = p_bayes;
                z = [z , obj.x0 + obj.sigma_z0 * randn];
                p_posteriori = BayesFilter.normGaussian(z(end), obj.sigma_z0, x);
                p_z = sum(p_posteriori .* p_priori * dx);
                p_bayes = (p_posteriori .* p_priori) / p_z;
                
                E_priori = sum(x .* p_priori * dx);
                max_priori = max(p_priori);
                max_posteriori = 1/sqrt(2 * pi * obj.sigma_z0^2);
                E_bayes = [E_bayes, sum(x .* p_bayes *dx)];
                var_bayes = sum((x-E_bayes(end)).^2 .* p_bayes * dx);
                sigma_bayes = [sigma_bayes, sqrt(var_bayes)];
                max_bayes = max(p_bayes);
                
                subplot(3,2,1); stem(obj.x0, max_priori, 'g', 'linewidth', 5); hold on;
                plot(x, p_priori, 'b', 'linewidth', 2);
                stem(E_priori, max_priori, 'b', 'linewidth', 2);
                hold off; xlabel('x'); ylabel('p_{priori}(x)'); grid on;
                axis([obj.x_ll obj.x_rl 0 max_priori]); title('FDP à priori')
                
                subplot(3,2,3); stem(obj.x0, max_posteriori, 'g', 'linewidth', 5); hold on;
                plot(x, max_posteriori, 'r', 'linewidth', 2);
                stem(z(end), max_posteriori, 'r', 'linewidth', 2);
                hold off; xlabel('x'); ylabel('p_{posteriori}(x)'); grid on;
                axis([obj.x_ll obj.x_rl 0 max_posteriori]); title('FDP à posteriori')
                
                subplot(3,2,5); stem(obj.x0, max_bayes, 'g', 'linewidth', 5); hold on;
                plot(x, p_bayes, 'k', 'linewidth', 2);
                stem(E_bayes(end), max_bayes, 'k', 'linewidth', 2);
                hold off; xlabel('x'); ylabel('p_{bayes}(x)'); grid on;
                axis([obj.x_ll obj.x_rl 0 max_bayes]); title('FDP Bayes')
                
                subplot(3,2,2);
                plot(1:k, z, 'r', 'linewidth', 2); hold on;
                plot(1:k, obj.x0 * ones(1, k), 'k', 'linewidth', 2);
                hold off; grid on; axis([0 k min([z obj.x0]) max([z obj.x0])]);
                xlabel('tempo [número de iterações]'); ylabel('sensor');
                title(sprintf('Medida do Sensor: %.4f', z(end)))
                
                subplot(3,2,4);
                plot(1:k, E_bayes, 'r', 'linewidth', 2); hold on;
                plot(1:k, obj.x0 * ones(1, k), 'k', 'linewidth', 2); hold on;
                hold off; grid on; axis([0 k min([E_bayes obj.x0]) max([E_bayes obj.x0])]);
                xlabel('tempo [número de iterações]'); ylabel('Esperança');
                title(sprintf('Medida do Bayes: %.4f', E_bayes(end)))
                
                plot(1:k, z, 'r', 'linewidth', 2); hold on;
                plot(1:k, obj.x0 * ones(1, k), 'k', 'linewidth', 2);
                hold off; grid on; axis([0 k min([z obj.x0]) max([z obj.x0])]);
                xlabel('tempo [número de iterações]'); ylabel('Sensor');
                title(sprintf('Medida do Sensor: %.4f', z(end)))
                
                subplot(3,2,4);
                plot(1:k, E_bayes, 'r', 'linewidth', 2); hold on;
                plot(1:k, obj.x0 * ones(1, k), 'k', 'linewidth', 2); hold on;
                hold off; grid on; axis([0 k min([E_bayes obj.x0]) max([E_bayes obj.x0])]);
                xlabel('tempo [número de iterações]'); ylabel('Esperança');
                title(sprintf('Medida do Bayes: %.4f', E_bayes(end)))
                
                subplot(3,2,6);
                plot(1:k, sigma_bayes, 'r', 'linewidth', 2); hold on;
                plot(1:k, stop_sigma * ones(1, k), 'r', 'linewidth', 2); hold off;
                grid on; axis([0 k 0 max(sigma_bayes)]);
                xlabel('tempo [número de iterações]'); ylabel('Desvio Padrão');
                title(sprintf('Sigma de Bayes: %.4f', sigma_bayes(end)));
                
                drawnow;
                
                
            end
            
        end
        
    end
    
    methods(Static)
       function f =  normGaussian(mu, sigma_z, x)
            f = 1/sqrt(2 * pi * sigma_z^2) * exp(-1/2 * (x-mu).^2 / sigma_z^2);
       end 
    end
end

