// var app = angular.module('RestApiTesterApp', ['ui.bootstrap'])
var app = angular.module('RestApiTesterApp', ['ui.bootstrap'])

app.controller("main", function($scope, $location, $http, $filter, $log) {
    var REQUEST_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS'];
    var REQUEST_PREFIX = '';

	var absUrl = $location.absUrl();
	// var restapiEndpointIndex = absUrl.indexOf('/resources/static/restapi-tester.html');
	var restapiEndpointIndex = absUrl.indexOf('/restapitester');
	// $scope.requestPath = $location.absUrl().slice(0, restapiEndpointIndex) + '/api/v1/todos/';
	$scope.requestPath = $location.absUrl().slice(0, restapiEndpointIndex) + '/most_similar?word=king&topn=10';
	
    $scope.requestPrefix = REQUEST_PREFIX;
    $scope.requestMethod = REQUEST_METHODS[0];
    // $scope.requestPath = '';
    // $scope.requestPath = 'http://localhost:8080/most_similar?word=king&topn=10';
    // $scope.requestBody = '{ "todoTitle": "Hello World!" }';
    $scope.requestBody = '';
    $scope.responseBody = '';
    $scope.responseHeaders = '';

    $scope.requestSubmitClick = function() {
        var conf = {};
        conf.method = $scope.requestMethod;
        conf.url = $scope.requestPrefix + $scope.requestPath;
        switch (conf.method) {
            case 'POST':
            case 'PUT':
                conf.data = $scope.requestBody;
        }

        $http(conf).success(function(data, status, headers, conf) {
            $scope.responseStatus = status;
            if (typeof data === 'string') {
                $scope.responseBody = data;
            } else {
                $scope.responseBody = $filter('json')(data);
            }
            $scope.responseHeaders = angular.toJson(headers(), true)
        }).error(function(data, status, headers, conf) {
            $scope.responseStatus = status;
            if (typeof data === 'string') {
                $scope.responseBody = data;
            } else {
                $scope.responseBody = $filter('json')(data);
            }
        });
    };

    $scope.requestPathChange = function() {
        if (($scope.requestPrefix + $scope.requestPath).length > 0) {
            $scope.submitDisabled = false;
        } else {
            $scope.submitDisabled = true;
        }
    };

    $scope.requestPathChange();
    
});
