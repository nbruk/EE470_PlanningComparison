%% run_comparison.m
% Path-planning comparison for autonomous driving simulation.
%
% This script compares:
%   1. A* on a waypoint graph
%   2. Hybrid A*
%   3. RRT* with multiple random seeds
%
% The planners are deployed on the same scenes.
% Automated Driving can show the Hybrid A* and best RRT* driving 
%
% Utilizes:
%   - Navigation Toolbox
%   - Automated Driving Toolbox

clear; clc; close all;

%% ===================== Settings ============================================

cfg = struct();

% --- Randomness --------------------------------------------------------------
cfg.rngSeed = 0;                     % repeatable RRT* results
cfg.numRRTSeeds = 25;                % number of RRT* trials

% --- Map settings ------------------------------------------------------------
cfg.mapWidth = 50;                   % meters
cfg.mapHeight = 50;                  % meters
cfg.mapResolution = 1;               % occupancy map cells per meter

% --- Evaluation settings -----------------------------------------------------
cfg.goalTolerance = 1.0;             % acceptable final position error in meters
cfg.timeBudget = 5;                  % max time allowed for RRT* run in seconds

% --- A* waypoint graph settings ---------------------------------------------
cfg.astarStep = 2;                   % graph nodes spacing in meters
cfg.edgeCheckStep = 0.5;             % collision-check spacing along graph edges

% --- Hybrid A* settings ------------------------------------------------------
cfg.hybridTurnRadius = 4;            % minimum turning radius in meters
cfg.hybridPrimitiveLength = 4;       % motion primitive length in meters

% --- RRT* settings -----------------------------------------------------------
cfg.rrtConnectionDistance = 5;       % max distance between connected states
cfg.rrtMaxIterations = 4000;         % max number of RRT* iterations
cfg.rrtMaxNodes = 8000;              % max number of tree nodes

% --- Automated Driving Toolbox playback -------------------------------------
cfg.useSimulation = true;            % toggle playback
cfg.simSampleTime = 0.1;             % seconds
cfg.simStopTime = 20;                % seconds
cfg.vehicleSpeed = 5;                % m/s

%% ===================== Setup ===============================================

rng(cfg.rngSeed);
rrtSeeds = randi(1e6, cfg.numRRTSeeds, 1);

scenes = makeScenes();
resultsRows = {};

%% ===================== Run each scene ======================================

for sceneIndex = 1:numel(scenes)

    scene = scenes(sceneIndex);

    fprintf('\nRunning scene: %s\n', scene.name);

    [map, validator, stateSpace] = makeEnvironment(scene, cfg);

    astarResult = planAStarGraph(map, scene, cfg);
    hybridResult = planHybridAStar(validator, scene, cfg);
    rrtResults = runRRTSeeds(validator, stateSpace, scene, cfg, rrtSeeds);

    resultsRows(end+1,:) = packRow(scene.name, 'A* waypoint', astarResult); %#ok<SAGROW>
    resultsRows(end+1,:) = packRow(scene.name, 'Hybrid A*', hybridResult);
    resultsRows(end+1,:) = packSeededRow(scene.name, 'RRT* median', rrtResults.all);

    plotPlannerResults(map, scene, sceneIndex, astarResult, hybridResult, rrtResults.best);

    if cfg.useSimulation
        runTwoCarSimulation(scene, hybridResult, rrtResults.best, cfg);
    end

end

%% ===================== Save results ========================================

resultsTable = cell2table(resultsRows, 'VariableNames', ...
    {'scene', 'planner', 'comp_time_s', 'path_len_m', ...
     'smoothness_sumdtheta', 'goal_pos_err_m'});

disp(resultsTable);
save('metrics.mat', 'resultsTable');

fprintf('\nSaved metrics.mat and %d scene figures.\n', numel(scenes));

%% ===================== Scene definitions ===================================

function scenes = makeScenes()
% Obstacles are stored as [x y width height].
% x,y represent the lower-left corner of the rectangle.

    scenes = struct('name', {}, 'start', {}, 'goal', {}, 'obs', {});

    scenes(1).name = 'narrow_passage';
    scenes(1).start = [5 25 0];
    scenes(1).goal = [48 25 0];
    scenes(1).obs = [15 10 5 12;
                     15 28 5 12;
                     40 10 5 11;
                     40 29 5 11;
                     28 20 5 10];

    scenes(2).name = 'zig_zag';
    scenes(2).start = [5 10 pi/2];
    scenes(2).goal = [45 40 0];
    scenes(2).obs = [15 0 6 35;
                     30 15 6 35];

    scenes(3).name = 'parked_vehicles';
    scenes(3).start = [5 25 0];
    scenes(3).goal = [45 25 0];
    scenes(3).obs = [14 18 6 4;
                     26 23 6 4;
                     38 14 6 4];

    scenes(4).name = 'bounded_object';
    scenes(4).start = [5 5 pi/2];
    scenes(4).goal = [45 45 0];
    scenes(4).obs = [22 22 6 6;
                     10 30 3 15;
                     37 5  3 15];

end

%% ===================== Environment setup ===================================

function [map, validator, stateSpace] = makeEnvironment(scene, cfg)

    map = binaryOccupancyMap(cfg.mapWidth, cfg.mapHeight, cfg.mapResolution);

    for obstacleIndex = 1:size(scene.obs, 1)
        addRectangleToMap(map, scene.obs(obstacleIndex,:), cfg);
    end

    stateSpace = stateSpaceSE2([0 cfg.mapWidth; 0 cfg.mapHeight; -pi pi]);

    validator = validatorOccupancyMap(stateSpace);
    validator.Map = map;
    validator.ValidationDistance = 0.5;

end

function addRectangleToMap(map, obstacle, cfg)

    xMin = max(0, obstacle(1));
    yMin = max(0, obstacle(2));
    xMax = min(cfg.mapWidth,  obstacle(1) + obstacle(3));
    yMax = min(cfg.mapHeight, obstacle(2) + obstacle(4));

    [xGrid, yGrid] = meshgrid(xMin:1/cfg.mapResolution:xMax, ...
                              yMin:1/cfg.mapResolution:yMax);

    setOccupancy(map, [xGrid(:), yGrid(:)], 1);

end

%% ===================== A* waypoint graph ===================================

function resultStruct = planAStarGraph(map, scene, cfg)

    startTimer = tic;

    nodes = makeWaypointNodes(map, cfg.astarStep);

    if isempty(nodes)
        resultStruct = failedResult(toc(startTimer));
        return;
    end

    [sourceNodes, targetNodes, edgeWeights] = makeWaypointEdges(map, nodes, cfg);

    if isempty(sourceNodes)
        resultStruct = failedResult(toc(startTimer));
        return;
    end

    graphObject = graph(sourceNodes, targetNodes, edgeWeights, size(nodes, 1));

    startNode = nearestNode(nodes, scene.start(1:2));
    goalNode = nearestNode(nodes, scene.goal(1:2));

    nodePath = shortestpath(graphObject, startNode, goalNode);

    if isempty(nodePath)
        resultStruct = failedResult(toc(startTimer));
        return;
    end

    path = nodes(nodePath, :);
    goalError = norm(path(end,:) - scene.goal(1:2));

    resultStruct = makeResult( ...
        goalError <= cfg.goalTolerance * cfg.astarStep, ...
        toc(startTimer), ...
        path, ...
        goalError);

end

function nodes = makeWaypointNodes(map, step)

    mapWidth = map.XWorldLimits(2);
    mapHeight = map.YWorldLimits(2);

    [xGrid, yGrid] = meshgrid(0:step:mapWidth, 0:step:mapHeight);
    nodes = [xGrid(:), yGrid(:)];

    freeNodeMask = ~checkOccupancy(map, nodes);
    nodes = nodes(freeNodeMask, :);

end

function [sourceNodes, targetNodes, edgeWeights] = makeWaypointEdges(map, nodes, cfg)

    searchRadius = cfg.astarStep * 1.5;

    neighborSearcher = KDTreeSearcher(nodes);
    neighbors = rangesearch(neighborSearcher, nodes, searchRadius);

    sourceNodes = [];
    targetNodes = [];
    edgeWeights = [];

    for i = 1:numel(neighbors)
        for j = neighbors{i}

            if j <= i
                continue;
            end

            if isEdgeFree(map, nodes(i,:), nodes(j,:), cfg.edgeCheckStep)
                sourceNodes(end+1) = i; %#ok<AGROW>
                targetNodes(end+1) = j; %#ok<AGROW>
                edgeWeights(end+1) = norm(nodes(i,:) - nodes(j,:)); %#ok<AGROW>
            end

        end
    end

end

%% ===================== Hybrid A* ===========================================

function resultStruct = planHybridAStar(validator, scene, cfg)

    startTimer = tic;

    try
        planner = plannerHybridAStar(validator, ...
            'MinTurningRadius', cfg.hybridTurnRadius, ...
            'MotionPrimitiveLength', cfg.hybridPrimitiveLength);

        route = plan(planner, scene.start, scene.goal);

        path = route.States(:, 1:2);
        goalError = norm(path(end,:) - scene.goal(1:2));

        resultStruct = makeResult( ...
            true, ...
            toc(startTimer), ...
            path, ...
            goalError);

    catch
        resultStruct = failedResult(toc(startTimer));
    end

end

%% ===================== RRT* =================================================

function rrtResults = runRRTSeeds(validator, stateSpace, scene, cfg, rrtSeeds)

    rrtResults.all = repmat(failedResult(0), numel(rrtSeeds), 1);
    rrtResults.best = [];

    for seedIndex = 1:numel(rrtSeeds)

        singleResult = planSingleRRT(validator, stateSpace, scene, cfg, rrtSeeds(seedIndex));
        rrtResults.all(seedIndex) = singleResult;

        if singleResult.success && ...
                (isempty(rrtResults.best) || singleResult.path_length < rrtResults.best.path_length)
            rrtResults.best = singleResult;
        end

    end

end

function resultStruct = planSingleRRT(validator, stateSpace, scene, cfg, seed)

    startTimer = tic;
    rng(seed);

    try
        planner = plannerRRTStar(stateSpace, validator);
        planner.MaxConnectionDistance = cfg.rrtConnectionDistance;
        planner.MaxIterations = cfg.rrtMaxIterations;
        planner.MaxNumTreeNodes = cfg.rrtMaxNodes;

        [route, info] = plan(planner, scene.start, scene.goal);

        elapsedTime = toc(startTimer);

        if ~info.IsPathFound || elapsedTime > cfg.timeBudget
            resultStruct = failedResult(elapsedTime);
            return;
        end

        path = route.States(:, 1:2);
        goalError = norm(path(end,:) - scene.goal(1:2));

        resultStruct = makeResult( ...
            goalError <= cfg.goalTolerance, ...
            elapsedTime, ...
            path, ...
            goalError);

    catch
        resultStruct = failedResult(toc(startTimer));
    end

end

%% ===================== Results table helpers ================================

function row = packRow(sceneName, plannerName, resultStruct)

    row = {sceneName, plannerName, ...
           resultStruct.comp_time, resultStruct.path_length, ...
           resultStruct.smoothness, resultStruct.goal_err};

end

function row = packSeededRow(sceneName, plannerName, allResults)

    successMask = [allResults.success];

    if ~any(successMask)
        row = {sceneName, plannerName, NaN, NaN, NaN, NaN};
        return;
    end

    successfulResults = allResults(successMask);

    row = {sceneName, plannerName, ...
           median([successfulResults.comp_time]), ...
           median([successfulResults.path_length]), ...
           median([successfulResults.smoothness]), ...
           median([successfulResults.goal_err])};

end

%% ===================== Planner plotting ====================================

function plotPlannerResults(map, scene, sceneIndex, astarResult, hybridResult, rrtResult)

    figure('Name', scene.name, 'Color', 'w');
    show(map);
    hold on;

    legendHandles = gobjects(0);
    legendLabels = {};

    [legendHandles, legendLabels] = addToLegend(legendHandles, legendLabels, ...
        plot(scene.start(1), scene.start(2), 'go', ...
        'MarkerFaceColor', 'g', 'MarkerSize', 8), ...
        'start');

    [legendHandles, legendLabels] = addToLegend(legendHandles, legendLabels, ...
        plot(scene.goal(1), scene.goal(2), 'rp', ...
        'MarkerFaceColor', 'r', 'MarkerSize', 12), ...
        'goal');

    [legendHandles, legendLabels] = plotPathIfAvailable( ...
        legendHandles, legendLabels, astarResult, 'k--', 1.5, 'A* waypoint');

    [legendHandles, legendLabels] = plotPathIfAvailable( ...
        legendHandles, legendLabels, hybridResult, 'm-', 2.0, 'Hybrid A*');

    [legendHandles, legendLabels] = plotPathIfAvailable( ...
        legendHandles, legendLabels, rrtResult, 'b-', 2.0, 'best RRT*');

    title(sprintf('Scene %d: %s', sceneIndex, strrep(scene.name, '_', '\_')));
    legend(legendHandles, legendLabels, 'Location', 'best');

    formatAxes(gca, map.XWorldLimits(2), map.YWorldLimits(2));

    xlabel('x (m)');
    ylabel('y (m)');

    hold off;

end

function [legendHandles, legendLabels] = plotPathIfAvailable( ...
    legendHandles, legendLabels, resultStruct, lineStyle, lineWidth, labelText)

    if ~hasPath(resultStruct)
        return;
    end

    pathHandle = plot(resultStruct.path(:,1), resultStruct.path(:,2), ...
        lineStyle, 'LineWidth', lineWidth);

    [legendHandles, legendLabels] = addToLegend(legendHandles, legendLabels, ...
        pathHandle, labelText);

end

function [legendHandles, legendLabels] = addToLegend(legendHandles, legendLabels, plotHandle, labelText)

    legendHandles(end+1) = plotHandle;
    legendLabels{end+1} = labelText;

end

%% ===================== Automated Driving Toolbox simulation =================

function runTwoCarSimulation(scene, hybridResult, rrtResult, cfg)

    if ~hasPath(hybridResult) && ~hasPath(rrtResult)
        warning('No valid Hybrid A* or RRT* path for simulation: %s', scene.name);
        return;
    end

    scenario = drivingScenario( ...
        'SampleTime', cfg.simSampleTime, ...
        'StopTime', cfg.simStopTime);

    addScenarioObstacles(scenario, scene.obs);

    if hasPath(hybridResult)
        addVehicleOnPath(scenario, ...
            'HybridAStarVehicle', scene.start, hybridResult.path, ...
            cfg.vehicleSpeed, 0.5);
    end

    if hasPath(rrtResult)
        addVehicleOnPath(scenario, ...
            'RRTStarVehicle', scene.start, rrtResult.path, ...
            cfg.vehicleSpeed, -0.5);
    end

    figure('Name', ['Automated Driving Scenario: ', scene.name], 'Color', 'w');
    ax = axes;

    plot(scenario, 'Parent', ax);

    title(ax, ['Hybrid A* vs RRT* Playback: ', strrep(scene.name, '_', '\_')]);
    xlabel(ax, 'x (m)');
    ylabel(ax, 'y (m)');

    formatAxes(ax, cfg.mapWidth, cfg.mapHeight);

    while advance(scenario)
        formatAxes(ax, cfg.mapWidth, cfg.mapHeight);
        pause(cfg.simSampleTime);
    end

end

function addScenarioObstacles(scenario, obstacles)

    for obstacleIndex = 1:size(obstacles, 1)

        obstacle = obstacles(obstacleIndex,:);  % [x y w h]

        centerX = obstacle(1) + obstacle(3)/2;
        centerY = obstacle(2) + obstacle(4)/2;

        actor(scenario, ...
            'ClassID', 5, ...
            'Name', sprintf('Obstacle_%d', obstacleIndex), ...
            'Length', obstacle(3), ...
            'Width', obstacle(4), ...
            'Height', 2.0, ...
            'Position', [centerX, centerY, 1.0], ...
            'Yaw', 0);

    end

end

function addVehicleOnPath(scenario, vehicleName, startPose, path, speed, yOffset)

    car = vehicle(scenario, ...
        'ClassID', 1, ...
        'Name', vehicleName, ...
        'Length', 4.7, ...
        'Width', 1.8, ...
        'Height', 1.5, ...
        'Position', [startPose(1), startPose(2) + yOffset, 0], ...
        'Yaw', rad2deg(startPose(3)));

    waypoints = [path(:,1), path(:,2), zeros(size(path,1), 1)];
    speeds = speed * ones(size(waypoints, 1), 1);

    trajectory(car, waypoints, speeds);

end

%% ===================== Metrics and small utilities ==========================

function resultStruct = makeResult(success, compTime, path, goalError)
    resultStruct.success = success;
    resultStruct.comp_time = compTime;
    resultStruct.path_length = pathLength(path);
    resultStruct.smoothness = pathSmoothness(path);
    resultStruct.goal_err = goalError;
    resultStruct.path = path;
end

function resultStruct = failedResult(compTime)
    resultStruct.success = false;
    resultStruct.comp_time = compTime;
    resultStruct.path_length = NaN;
    resultStruct.smoothness = NaN;
    resultStruct.goal_err = NaN;
    resultStruct.path = [];
end

function tf = hasPath(result)
    tf = ~isempty(result) && ...
         isfield(result, 'path') && ...
         ~isempty(result.path);
end

function tf = isEdgeFree(map, startPoint, endPoint, checkStep)
    numPoints = max(2, ceil(norm(endPoint - startPoint) / checkStep));
    points = [linspace(startPoint(1), endPoint(1), numPoints)', ...
              linspace(startPoint(2), endPoint(2), numPoints)'];
    tf = ~any(checkOccupancy(map, points));
end

function index = nearestNode(nodes, point)
    [~, index] = min(vecnorm(nodes - point, 2, 2));
end

function lengthMeters = pathLength(path)
    lengthMeters = sum(vecnorm(diff(path), 2, 2));
end

function smoothnessValue = pathSmoothness(path)
    if size(path, 1) < 3
        smoothnessValue = 0;
        return;
    end
    headings = atan2(diff(path(:,2)), diff(path(:,1)));
    headingChanges = abs(wrapToPi(diff(headings)));
    smoothnessValue = sum(headingChanges);
end

function formatAxes(ax, width, height)
    grid(ax, 'on');
    grid(ax, 'minor');
    xlim(ax, [0 width]);
    ylim(ax, [0 height]);
    ax.XTick = 0:5:width;
    ax.YTick = 0:5:height;
    ax.Layer = 'top';
    axis(ax, 'equal');
    view(ax, 2);

end