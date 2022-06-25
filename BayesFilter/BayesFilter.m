classdef BayesFilter < handle
    properties
        x_ll
        x_rl
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
        
        function [x, dx] = calcInitialPositions(obj)
            dx = (obj.x_rl - obj.x_ll)/obj.spaces;
            x = obj.x_ll:dx:obj.x_rl;
        end
        
        function z = generateRandomMeasure(obj)
            z = obj.x0 + obj.sigma_z0 * randn;
        end
        
        function p = generateLinearProbability(obj)
            [x, ~] = obj.calcInitialPositions();
            p = 1/(obj.x_rl - obj.x_ll) * ones(size(x));
        end
        
        function p_z = calcIntegrateMeasure(obj, p_priori, p_posteriori)
            [~, dx] = obj.calcInitialPositions();
            p_z = sum(p_posteriori .* p_priori * dx);
        end
        
        function f =  normGaussian(obj, mu, sigma_z)
           [x, ~] = obj.calcInitialPositions();
           f = 1/sqrt(2 * pi * sigma_z^2) * exp(-1/2 * (x-mu).^2 / sigma_z^2);
        end

        function E = calcIntegrateExpectancy(obj, p)
           [x, dx] = obj.calcInitialPositions();
           E = sum(x .* p * dx);
        end

        function var = calcVariance(obj, mu, p)
            [x, dx] = obj.calcInitialPositions();
            var = sum((x-mu).^2 .* p * dx);
        end
        
        function [E, sigma] = getMoments(obj, p)
            E = obj.calcIntegrateExpectancy(p);
            var = obj.calcVariance(E, p);
            sigma = sqrt(var);
        end
        
        function runSimulation(obj, confidence_percentage)
            
            obj.setRandom();

            z = obj.generateRandomMeasure();

            p_priori = obj.generateLinearProbability();
            p_posteriori = obj.normGaussian(z, obj.sigma_z0);
            
            p_z = obj.calcIntegrateMeasure(p_priori, p_posteriori);
            p_bayes = BayesFilter.calcBayesProbability(p_priori, p_posteriori, p_z);
            
            k = 1;
            
            stop_sigma = confidence_percentage * obj.sigma_z0;
            
            [E_bayes, sigma_bayes] = obj.getMoments(p_bayes);
            
            while sigma_bayes > stop_sigma
                k = k + 1;
                p_priori = p_bayes;
                
                z = obj.generateRandomMeasure();
                p_posteriori = obj.normGaussian(z, obj.sigma_z0);
                p_z = obj.calcIntegrateMeasure(p_priori, p_posteriori);
                p_bayes = BayesFilter.calcBayesProbability(p_priori, p_posteriori, p_z);
                
                [E_bayes, sigma_bayes] = obj.getMoments(p_bayes);
                
                obj.showCharts();
            end
        end
        
        function showProbabilityFunction(obj, p, chart_name, plot_id, color, E)
            [x, ~] = obj.calcInitialPositions();
            peak = max(p);
            
            subplot(3,2,plot_id); stem(obj.x0, peak, 'g', 'linewidth', 5); hold on;
            plot(x, p, color, 'linewidth', 2);
            stem(E, peak, color, 'linewidth', 2);
            hold off; xlabel('x'); ylabel('p(x)'); grid on;
            axis([obj.x_ll obj.x_rl 0 peak]); title(chart_name)
        end
        
        function showParamChat(~, chart_name, plot_id, param, ...
                color, inf_limit, sup_limit, param_name, param_reference, k)
            subplot(3,2,plot_id);
            plot(1:k, param, color, 'linewidth', 2); hold on;
            plot(1:k, param_reference * ones(1, k), color, 'linewidth', 2);
            hold off; grid on; axis([0 k inf_limit sup_limit]);
            xlabel('tempo [número de iterações]'); ylabel(param_name);
            title(sprintf('%s: %.4f', chart_name, param(end)))
        end
        
        function showCharts(obj)
            
            obj.showProbabilityFunction(p_priori, 'FDP à priori', 1, 'b', E_priori)
            obj.showProbabilityFunction(p_posteriori, 'FDP à posteriori', 3, 'r', z)
            obj.showProbabilityFunction(p_bayes, 'FDP Bayes', 5, 'k', E_bayes)
            
            obj.showParamChat('Medida do Sensor', 2, z, 'k', min([z obj.x0]), ...
                max([z obj.x0]), 'sendor', k, obj.x0);
            obj.showParamChat('Medida do Bayes', 4, E_bayes, 'r', min([E_bayes obj.x0]), ...
                max(sigma_bayes), 'esperança', k, obj.x0);
            obj.showParamChat('Sigma de Bayes', 6, sigma_bayes, 'r', 0, ...
                max([z obj.x0]), 'desvio padrão', k, stop_sigma);

            drawnow;
        end
        
    end
    
    methods(Static)
       function p_estimated = calcBayesProbability(p_priori, p_posteriori, p_measure)
           p_estimated = (p_posteriori .* p_priori) / p_measure;
       end
    end
end

