window.addEventListener('load', function() {
    // Change play rate of poster video.
    document.getElementById('poster-vid').playbackRate = 1.5;

    let channelImg = document.getElementById('channel-img');

    let prevActive = null;
    function setActive(newElem) {
        if (prevActive) {
            prevActive.classList.toggle('active');
        }
        prevActive = newElem;
        newElem.classList.toggle('active');
    }

    let visiblityChannel = document.getElementById('visibility-channel');
    let landChannel = document.getElementById('land-channel');
    let foodChannel = document.getElementById('food-channel');
    let antHillChannel = document.getElementById('ant-hill-channel');
    let playerAntChannel = document.getElementById('player-ant-channel');
    let enemyAntChannel = document.getElementById('enemy-ant-channel');

    setActive(visiblityChannel);
    visiblityChannel.addEventListener('click', e => {
        channelImg.src = "assets/images/visibility.png";
        setActive(visiblityChannel);
    });

    landChannel.addEventListener('click', e => {
        channelImg.src = "assets/images/land_water.png";
        setActive(landChannel);
    });

    foodChannel.addEventListener('click', e => {
        channelImg.src = "assets/images/food.png";
        setActive(foodChannel);
    });

    antHillChannel.addEventListener('click', e => {
        channelImg.src = "assets/images/anthills.png";
        setActive(antHillChannel);
    });

    playerAntChannel.addEventListener('click', e => {
        channelImg.src = "assets/images/player_ant.png";
        setActive(playerAntChannel);
    });

    enemyAntChannel.addEventListener('click', e => {
        channelImg.src = "assets/images/enemy_ant.png";
        setActive(enemyAntChannel);
    });
});