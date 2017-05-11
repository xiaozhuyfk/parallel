root = exports ? this


http =

    get: (options, callback)->
        options.method = 'get'
        @request options, callback

    post: (options, callback)->
        options.method = 'post'
        @request options, callback

    request: (options, callback)->
        method = options.method?.toLowerCase() or 'get'
        url = options.url
        dataType = options.dataType?.toLowerCase()
        data = options.data
        unless data instanceof FormData
            if dataType is 'json'
                contentType = 'application/json;charset=UTF-8'
            else
                contentType = 'application/x-www-form-urlencoded;charset=UTF-8'
        if data?
            if contentType? and typeof data is 'object'
                switch dataType
                    when 'json'
                        data = JSON.stringify data
                    else
                        data = querystring.stringify data
            if method is 'get'
                url = "#{url}?#{data}"
                data = null

        xhr = new (window.ActiveXObject or XMLHttpRequest)('Microsoft.XMLHTTP')
        if options.onProgress?
            xhr.upload.onprogress = (e)->
                if e.lengthComputable
                    options.onProgress e.loaded / e.total
        if options.onComplete?
            xhr.upload.onload = (e)->
                options.onComplete()
        if callback?
            xhr.onreadystatechange = (e)->
                if xhr.readyState is 4
                    if xhr.status in [0, 200]
                        data = xhr.responseText
                        try
                            data = JSON.parse data
                        catch err
                            try
                                data = querystring.parse data
                            catch err
                        callback null, data
                    else
                        callback
                            code: xhr.status
                            message: "#{xhr.status} (#{xhr.statusText})" + if xhr.responseText then ": #{xhr.responseText}" else ''
        xhr.open method, url, true
        xhr.overrideMimeType? 'text/plain'
        xhr.setRequestHeader 'X-Requested-With', 'XMLHttpRequest'
        if contentType then xhr.setRequestHeader 'content-type', contentType
        xhr.send data

# Help with the placement of nodes
RadialPlacement = () ->
    # stores the key -> location values
    values = d3.map()
    # how much to separate each location by
    increment = 20
    # how large to make the layout
    radius = 200
    # where the center of the layout should be
    center = {"x": 0, "y": 0}
    # what angle to start at
    start = -120
    current = start

    # Given an center point, angle, and radius length,
    # return a radial position for that angle
    radialLocation = (center, angle, radius) ->
        x = (center.x + radius * Math.cos(angle * Math.PI / 180))
        y = (center.y + radius * Math.sin(angle * Math.PI / 180))
        {"x": x, "y": y}

    # Main entry point for RadialPlacement
    # Returns location for a particular key,
    # creating a new location if necessary.
    placement = (key) ->
        value = values.get(key)
        if !values.has(key)
            value = place(key)
        value

    # Gets a new location for input key
    place = (key) ->
        value = radialLocation(center, current, radius)
        values.set(key, value)
        current += increment
        value

    # Given a set of keys, perform some
    # magic to create a two ringed radial layout.
    # Expects radius, increment, and center to be set.
    # If there are a small number of keys, just make
    # one circle.
    setKeys = (keys) ->
        # start with an empty values
        values = d3.map()

        # number of keys to go in first circle
        firstCircleCount = 360 / increment

        # if we don't have enough keys, modify increment
        # so that they all fit in one circle
        if keys.length < firstCircleCount
            increment = 360 / keys.length

        # set locations for inner circle
        firstCircleKeys = keys.slice(0, firstCircleCount)
        firstCircleKeys.forEach (k) -> place(k)

        # set locations for outer circle
        secondCircleKeys = keys.slice(firstCircleCount)

        # setup outer circle
        radius = radius + radius / 1.8
        increment = 360 / secondCircleKeys.length

        secondCircleKeys.forEach (k) -> place(k)

    placement.keys = (_) ->
        if !arguments.length
            return d3.keys(values)
        setKeys(_)
        placement

    placement.center = (_) ->
        if !arguments.length
            return center
        center = _
        placement

    placement.radius = (_) ->
        if !arguments.length
            return radius
        radius = _
        placement

    placement.start = (_) ->
        if !arguments.length
            return start
        start = _
        current = start
        placement

    placement.increment = (_) ->
        if !arguments.length
            return increment
        increment = _
        placement

    return placement

graph = null
top5 = null

Network = () ->
    width = 1440
    height = 960

    # allData will store the unfiltered data
    allData = []
    curLinksData = []
    curNodesData = []
    linkedByIndex = {}

    # these will hold the svg groups for
    # accessing the nodes and links display
    vis = null
    nodesG = null
    linksG = null

    # these will point to the circles and lines
    # of the nodes and links
    node = null
    link = null

    # variables to refect the current settings
    # of the visualization
    layout = "force"
    filter = "all"

    # our force directed layout
    force = d3.layout.force()

    # color function used to color nodes
    nodeColors = d3.scale.category20()

    # tooltip used to display details
    tooltip = Tooltip("vis-tooltip", 230)

    # Starting point for network visualization
    # Initializes visualization and starts force layout
    network = (selection, data) ->
        # format our data
        graph = data.graph
        top5 = data.top5

        margin = {top : -5, right : -5, bottom : -5, left : -5}

        allData = setupData(data.graph)
        zoom = d3.behavior.zoom().scaleExtent([1, 1]).on("zoom", zoomed)

        # create our svg and groups
        vis = d3.select(selection).append("svg")
            .attr("width", width)
            .attr("height", height)
            .attr("transform", "translate(" + margin.left + "," + margin.right + ")")
            #.call(zoom)

        linksG = vis.append("svg").append("g").attr("id", "links")
        nodesG = vis.append("svg").append("g").attr("id", "nodes")

        # setup the size of the force environment
        force.size([width, height])

        setLayout("force")
        setFilter("all")

        # perform rendering and start force layout
        update()

    zoomed = () ->
        vis.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")")

    # The update() function performs the bulk of the
    # work to setup our visualization based on the
    # current layout/sort/filter.
    #
    # update() is called everytime a parameter changes
    # and the network needs to be reset.
    update = () ->
        # filter data to show based on current filter settings.
        curNodesData = filterNodes(allData.nodes)
        curLinksData = filterLinks(allData.links, curNodesData)

        # reset nodes in force layout
        force.nodes(curNodesData)

        # enter / exit for nodes
        updateNodes()

        # always show links in force layout
        force.links(curLinksData)
        updateLinks()

        # start me up!
        force.start()

    network.updateLayout = (newlayout) ->
        layout = newlayout
        if layout == "force"
            allData = setupData(graph)
            link.remove()
            node.remove()
            update()
        else
            allData = setupData(top5)
            link.remove()
            node.remove()
            update()


    # Public function to update highlighted nodes
    # from search
    network.updateSearch = (searchTerm) ->
        http.get {
            url: 'https://ec2-54-165-176-51.compute-1.amazonaws.com:443/network/' + searchTerm,
            dataType: 'json'
        }, (code, data) ->

            if typeof data.graph isnt 'undefined'
                graph = data.graph
                top5 = data.top5
            else
                empty = {
                    nodes : [
                        {
                            artist : "",
                            id : "EMPTY",
                            match : 1.0,
                            name : "Sorry, I don't know the answer.",
                            playcount : 10
                        }
                    ],
                    links : []
                }
                graph = empty
                top5 = empty

            if layout == "force"
                allData = setupData(graph)
                link.remove()
                node.remove()
                update()
            else
                if layout == "top5"
                    allData = setupData(top5)
                    link.remove()
                    node.remove()
                    update()


    network.updateData = (newData) ->
        allData = setupData(newData)
        link.remove()
        node.remove()
        update()

    # called once to clean up raw data and switch links to
    # point to node instances
    # Returns modified data
    setupData = (data) ->
        # initialize circle radius scale
        countExtent = d3.extent(data.nodes, (d) -> d.playcount)
        circleRadius = d3.scale.sqrt().range([3, 12]).domain(countExtent)

        data.nodes.forEach (n) ->
            # set initial x/y to values within the width/height
            # of the visualization
            n.x = randomnumber = Math.floor(Math.random() * width)
            n.y = randomnumber = Math.floor(Math.random() * height)
            n.radius = circleRadius(n.playcount)

        # id's -> node objects
        nodesMap = mapNodes(data.nodes)

        # switch links to point to node objects instead of id's
        data.links.forEach (l) ->

            if not l.source.id
              l.source = nodesMap.get(l.source)
              l.target = nodesMap.get(l.target)

            # linkedByIndex is used for link sorting
            linkedByIndex["#{l.source.id},#{l.target.id}"] = 1

        data

    # Helper function to map node id's to node objects.
    # Returns d3.map of ids -> nodes
    mapNodes = (nodes) ->
        nodesMap = d3.map()
        nodes.forEach (n) ->
            nodesMap.set(n.id, n)
        nodesMap

    # Removes nodes from input array
    # based on current filter setting.
    # Returns array of nodes
    filterNodes = (allNodes) ->
        filteredNodes = allNodes
        if filter == "popular" or filter == "obscure"
            playcounts = allNodes.map((d) -> d.playcount).sort(d3.ascending)
            cutoff = d3.quantile(playcounts, 0.5)
            filteredNodes = allNodes.filter (n) ->
                if filter == "popular"
                    n.playcount > cutoff
                else if filter == "obscure"
                    n.playcount <= cutoff

        filteredNodes

    # Removes links from allLinks whose
    # source or target is not present in curNodes
    # Returns array of links
    filterLinks = (allLinks, curNodes) ->
        curNodes = mapNodes(curNodes)
        allLinks.filter (l) ->
            curNodes.get(l.source.id) and curNodes.get(l.target.id)

    # enter/exit display for nodes
    updateNodes = () ->
        node = nodesG.selectAll("circle.node")
            .data(curNodesData, (d) -> d.id)

        node.enter().append("circle")
            .attr("class", "node")
            .attr("cx", (d) -> d.x)
            .attr("cy", (d) -> d.y)
            .attr("r", (d) -> d.radius)
            .style("fill", (d) -> nodeColors(d.artist))
            .style("stroke", (d) -> strokeFor(d))
            .style("stroke-width", 1.0)
            .call(force.drag)

        node.on("mouseover", showDetails)
            .on("mouseout", hideDetails)

        node.exit().remove()

    # enter/exit display for links
    updateLinks = () ->
        link = linksG.selectAll("line.link")
            .data(curLinksData, (d) -> "#{d.source.id}_#{d.target.id}")
        link.enter().append("line")
            .attr("class", "link")
            .attr("stroke", "#ddd")
            .attr("stroke-opacity", 0.8)
            .attr("x1", (d) -> d.source.x)
            .attr("y1", (d) -> d.source.y)
            .attr("x2", (d) -> d.target.x)
            .attr("y2", (d) -> d.target.y)

        link.exit().remove()

    # switches force to new layout parameters
    setLayout = (newLayout) ->
        force.on("tick", forceTick)
            .charge(-200)
            .linkDistance(50)

    # switches filter option to new filter
    setFilter = (newFilter) ->
        filter = newFilter


    # tick function for force directed layout
    forceTick = (e) ->
        node
            .attr("cx", (d) -> d.x)
            .attr("cy", (d) -> d.y)

        link
            .attr("x1", (d) -> d.source.x)
            .attr("y1", (d) -> d.source.y)
            .attr("x2", (d) -> d.target.x)
            .attr("y2", (d) -> d.target.y)


    # Helper function that returns stroke color for
    # particular node.
    strokeFor = (d) ->
        d3.rgb(nodeColors(d.artist)).darker().toString()

    neighboring = (a, b) ->
        linkedByIndex[a.id + "," + b.id] or
            linkedByIndex[b.id + "," + a.id]

    # Mouseover tooltip function
    showDetails = (d, i) ->
        content = '<p class="main">' + d.name + '</span></p>'
        content += '<hr class="tooltip-hr">'
        content += '<p class="main">' + d.artist + '</span></p>'
        tooltip.showTooltip(content, d3.event)

        # higlight connected links
        if link
            link.attr("stroke", (l) ->
                if l.source == d or l.target == d then "#555" else "#ddd"
            )
            .attr("stroke-opacity", (l) ->
                if l.source == d or l.target == d then 1.0 else 0.5
            )

        # highlight neighboring nodes
        # watch out - don't mess with node if search is currently matching
        node.style("stroke", (n) ->
            if (n.searched or neighboring(d, n)) then "#555" else strokeFor(n))
            .style("stroke-width", (n) ->
                if (n.searched or neighboring(d, n)) then 2.0 else 1.0)

        # highlight the node being moused over
        d3.select(this).style("stroke", "black")
            .style("stroke-width", 2.0)

    # Mouseout function
    hideDetails = (d, i) ->
        tooltip.hideTooltip()
        # watch out - don't mess with node if search is currently matching
        node.style("stroke", (n) -> if !n.searched then strokeFor(n) else "#555")
            .style("stroke-width", (n) -> if !n.searched then 1.0 else 2.0)
        if link
            link.attr("stroke", "#ddd")
                .attr("stroke-opacity", 0.8)

    # Final act of Network() function is to return the inner 'network()' function.
    return network

# Activate selector button
activate = (group, link) ->
    d3.selectAll("##{group} a").classed("active", false)
    d3.select("##{group} ##{link}").classed("active", true)

$ ->
    myNetwork = Network()

    d3.selectAll("#layouts a").on "click", (d) ->
        newLayout = d3.select(this).attr("id")
        activate("layouts", newLayout)
        myNetwork.updateLayout(newLayout)

    $("#song_select").on "change", (e) ->
        songFile = $(this).val()
        d3.json "data/#{songFile}", (json) ->
            myNetwork.updateData(json)

    $("#search").keyup (e) ->
        if e.which is 13
            searchTerm = $(this).val()
            myNetwork.updateSearch(searchTerm)

    d3.json "data/bigos.json", (json) ->
        myNetwork("#vis", json)
